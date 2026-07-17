// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/fused_butina.h"
#include "src/utils/cub_helpers.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"
#include "src/utils/nvtx.h"

namespace nvMolKit {
namespace {

// TODO: Profile cuda::memcpy_async (cp.async) for staging fingerprint word tiles in shared memory during the initial
// all-pairs count. Keep the current synchronous loads unless asynchronous copies provide a measurable speedup.
//
// ! Only supports fingerprinted clustering distance/similarity metrics and only for Tanimoto and Cosine similary
// currently
//
// speed gains mostly from pytorhc indexing + python type checking + reference checkiung + expensive loops, no cpu-gpu
// synchronization (CUDA Graphs)

// Launch geometry is kept explicit because the initial all-pairs pass and the iterative row kernels have different
// access patterns. The pair kernel assigns one candidate fingerprint to each thread, while the row kernels assign one
// fingerprint to each thread.
constexpr int kInitialArgMaxBlockSize = 256;
constexpr int kCandidateTileSize      = 128;
constexpr int kWordTileSize           = 8;
constexpr int kIterationBlockSize     = 128;

// Pack the neighbor count and original index into a 64 bit int to later perform comparisons to determine centroids
__device__ __forceinline__ std::uint64_t makeCandidate(int value, int index) {
  if (value < 0) {
    return 0;
  }

  // Since we are dealing with bits and shifts, unsafe to deal with signed ints. We will just take the -1 returned by
  // already clustered molecules and return 0. This means we need to increment all other values though.
  const auto encodedValue = static_cast<std::uint32_t>(value) + 1U;
  return (static_cast<std::uint64_t>(encodedValue) << 32) | static_cast<std::uint32_t>(index);
}

// store a candidate moleucle as the current maximum based on (value, index) pairs
__device__ __forceinline__ int storeCandidate(std::uint64_t candidate, int* maxValue, int* maxIndex) {
  const auto encodedValue = static_cast<std::uint32_t>(candidate >> 32);
  *maxValue               = encodedValue == 0 ? -1 : static_cast<int>(encodedValue) - 1;
  *maxIndex               = encodedValue == 0 ? -1 : static_cast<int>(static_cast<std::uint32_t>(candidate));
  return *maxValue;
}

// Apply the similarity threshold to an already computed fingerprint intersection. Only support tanimoto and cosine
// similartity for now
template <FingerprintSimilarityMetric Metric>
__device__ __forceinline__ bool intersectionMatches(int   intersection,
                                                    int   lhsBitCount,
                                                    int   rhsBitCount,
                                                    float threshold) {
  if constexpr (Metric == FingerprintSimilarityMetric::Tanimoto) {
    const int denominator = lhsBitCount + rhsBitCount - intersection;
    return denominator > 0 && static_cast<float>(intersection) >= threshold * static_cast<float>(denominator);
  } else if constexpr (Metric == FingerprintSimilarityMetric::Cosine) {
    const float denominator = sqrtf(static_cast<float>(lhsBitCount) * static_cast<float>(rhsBitCount));
    return denominator > 0.0F && static_cast<float>(intersection) >= threshold * denominator;
  } else {
    static_assert(Metric == FingerprintSimilarityMetric::Tanimoto || Metric == FingerprintSimilarityMetric::Cosine,
                  "Unsupported fingerprint similarity metric");
    return false;
  }
}

template <FingerprintSimilarityMetric Metric>
__device__ __forceinline__ bool fingerprintsWithinThreshold(const std::uint32_t* lhs,
                                                            const std::uint32_t* rhs,
                                                            int                  lhsBitCount,
                                                            int                  rhsBitCount,
                                                            int                  numWords,
                                                            float                threshold) {
  int intersection = 0;
  for (int word = 0; word < numWords; ++word) {
    intersection += __popc(lhs[word] & rhs[word]);
  }

  return intersectionMatches<Metric>(intersection, lhsBitCount, rhsBitCount, threshold);
}

// Compute and save each fingerprint's set-bit count. Both similarity metrics reuse these counts over and over again.
__global__ void fingerprintBitCountKernel(const cuda::std::span<const std::uint32_t> fingerprints,
                                          const cuda::std::span<int>                 bitCounts,
                                          int                                        numWords) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= static_cast<int>(bitCounts.size())) {
    return;
  }
  int bitCount = 0;
  for (int word = 0; word < numWords; ++word) {
    bitCount += __popc(fingerprints[row * numWords + word]);
  }
  bitCounts[row] = bitCount;
}

// Compute the initial active-neighbor count for every fingerprint without storing an N x N matrix.

//  Each molecule will be assigned a block with some threads (128 for now). Then each threads will compute whether the
// current molecule is a neighbor of the other molecules. Note: we are tiling in 2 dimensions, 1) molecules, 2) words
// for each fingerprint
template <FingerprintSimilarityMetric Metric>
__global__ void initialNeighborCountKernel(const cuda::std::span<const std::uint32_t> fingerprints,
                                           const cuda::std::span<const int>           bitCounts,
                                           const cuda::std::span<int>                 neighborCounts,
                                           int                                        numWords,
                                           float                                      threshold) {
  const int n   = static_cast<int>(neighborCounts.size());
  const int row = blockIdx.x;
  if (row >= n) {
    return;
  }
  __shared__ std::uint32_t centerWords[kWordTileSize];                          // current centroid words
  __shared__ std::uint32_t candidateWords[kCandidateTileSize * kWordTileSize];  // other molecules words

  int localNeighborCount = 0;

  // We now processes similarity between centroid and all other molecules, we tile over other molecules
  for (int candidateStart = 0; candidateStart < n; candidateStart += kCandidateTileSize) {
    const int candidateRow = candidateStart + threadIdx.x;
    int       intersection = 0;
    // We now have to loop over every bit in the fingerprint of this centroid vs another molecule, we tile over
    // fingerprint words
    for (int wordStart = 0; wordStart < numWords; wordStart += kWordTileSize) {
      // we use the first threads to load the current segment of the centroids fingerprint
      if (threadIdx.x < kWordTileSize) {
        const int word           = wordStart + threadIdx.x;
        centerWords[threadIdx.x] = word < numWords ? fingerprints[row * numWords + word] : 0;
      }
      // now we can load the other molecules current segment
      for (int item = threadIdx.x; item < kCandidateTileSize * kWordTileSize; item += blockDim.x) {
        const int tileRow    = item / kWordTileSize;
        const int tileWord   = item % kWordTileSize;
        const int sourceRow  = candidateStart + tileRow;
        const int sourceWord = wordStart + tileWord;
        candidateWords[item] =
          sourceRow < n && sourceWord < numWords ? fingerprints[sourceRow * numWords + sourceWord] : 0;
      }
      // wait for the current centroid segment to be loaded into memory
      __syncthreads();
      if (candidateRow < n) {
        for (int word = 0; word < kWordTileSize; ++word) {
          intersection += __popc(centerWords[word] & candidateWords[threadIdx.x * kWordTileSize + word]);
        }
      }
      // wait so that a thread doesn't erase a segment from memory that another thread is reading (centroid segments to
      // be precise)
      __syncthreads();
    }

    if (candidateRow < n) {
      localNeighborCount +=
        intersectionMatches<Metric>(intersection, bitCounts[row], bitCounts[candidateRow], threshold);
    }
  }

  // now we do a final reduce to get all the intersections for the current centroid
  __shared__ typename cub::BlockReduce<int, kCandidateTileSize>::TempStorage storage;
  const int neighborCount = cub::BlockReduce<int, kCandidateTileSize>(storage).Reduce(localNeighborCount, cubSum());
  if (threadIdx.x == 0) {
    neighborCounts[row] = neighborCount;
  }
}

// Select the initial centroid, We will use argmax kernel over neighborCounts
__global__ void initialArgMaxKernel(const cuda::std::span<const int> neighborCounts, int* maxValue, int* maxIndex) {
  std::uint64_t candidate = 0;
  // we tile over neighbors here
  for (int row = threadIdx.x; row < static_cast<int>(neighborCounts.size()); row += blockDim.x) {
    candidate = max(candidate, makeCandidate(neighborCounts[row], row));
  }

  // reduce the results of the tiling
  __shared__ typename cub::BlockReduce<std::uint64_t, kInitialArgMaxBlockSize>::TempStorage storage;
  candidate = cub::BlockReduce<std::uint64_t, kInitialArgMaxBlockSize>(storage).Reduce(candidate, cubMax());
  if (threadIdx.x == 0) {
    storeCandidate(candidate, maxValue, maxIndex);
  }
}

// Compare the selected centroid against every active molecule and extract all of its neighbors into the current
// cluster.
template <FingerprintSimilarityMetric Metric>
__global__ void extractClusterKernel(const cuda::std::span<const std::uint32_t> fingerprints,
                                     const cuda::std::span<const int>           bitCounts,
                                     const cuda::std::span<std::uint8_t>        active,
                                     const cuda::std::span<int>                 clusterMembers,
                                     int*                                       outputCount,
                                     const int*                                 maxValue,
                                     const int*                                 maxIndex,
                                     int                                        numWords,
                                     float                                      threshold) {
  // Ignore threads outside the molecule array, molecules already assigned to a cluster, and centroids with only 1
  // neighbor (themselves)
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= static_cast<int>(active.size()) || !active[row] || *maxValue <= 1) {
    return;
  }

  // Compare this active molecule with the selected centroid. Determine if thier fingerprints are within a similariy
  // threshold
  const int center = *maxIndex;
  if (!fingerprintsWithinThreshold<Metric>(fingerprints.data() + center * numWords,
                                           fingerprints.data() + row * numWords,
                                           bitCounts[center],
                                           bitCounts[row],
                                           numWords,
                                           threshold)) {
    return;
  }

  // Need an atomic here since we can get several matches at the same time.
  active[row]                = 0;
  const int outputSlot       = atomicAdd(outputCount, 1);
  clusterMembers[outputSlot] = row;
}

// Record the centroid and the offset of that centroid's cluster members in the big clusterMembers arr.
__global__ void recordClusterKernel(const int*                 maxValue,
                                    const int*                 maxIndex,
                                    const int*                 outputCount,
                                    int*                       clusterCount,
                                    const cuda::std::span<int> clusterOffsets,
                                    const cuda::std::span<int> centroids) {
  // A maximum of one means that every remaining molecule only neighbors itself, so no non-singleton cluster was
  // extracted.
  if (*maxValue <= 1) {
    return;
  }
  // Reserve the next cluster slot, save its centroid, and use the current output count as it's offset
  const int cluster           = atomicAdd(clusterCount, 1);
  centroids[cluster]          = *maxIndex;
  clusterOffsets[cluster + 1] = *outputCount;
}

// Update every active molecule's neighbor count after removing the current cluster, then produce one maximum candidate
// per block.
template <FingerprintSimilarityMetric Metric>
__global__ void updateNeighborCountsAndBlockMaxKernel(const cuda::std::span<const std::uint32_t> fingerprints,
                                                      const cuda::std::span<const int>           bitCounts,
                                                      const cuda::std::span<const std::uint8_t>  active,
                                                      const cuda::std::span<int>                 neighborCounts,
                                                      const cuda::std::span<const int>           clusterMembers,
                                                      const int*                                 clusterStart,
                                                      const int*                                 outputCount,
                                                      const int*                                 maxIndex,
                                                      const cuda::std::span<std::uint64_t>       blockMaxima,
                                                      int                                        numWords,
                                                      float                                      threshold) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;

  int updatedCount = -1;
  if (row < static_cast<int>(active.size()) && active[row]) {
    int decrement = 0;
    // Only compare against members added to the current cluster in the current iteration step.
    for (int memberSlot = *clusterStart; memberSlot < *outputCount; ++memberSlot) {
      const int removed = clusterMembers[memberSlot];
      // If we are here (active molecule) and we are comparing to the centroid, extractClusterKernel ran previously, and
      // it must not have found this molecule to be within threshold
      if (removed == *maxIndex) {
        continue;
      }
      decrement += fingerprintsWithinThreshold<Metric>(fingerprints.data() + row * numWords,
                                                       fingerprints.data() + removed * numWords,
                                                       bitCounts[row],
                                                       bitCounts[removed],
                                                       numWords,
                                                       threshold);
    }
    // Remove the matching rows from this molecule's stored active-neighbor count.
    updatedCount        = neighborCounts[row] - decrement;
    neighborCounts[row] = updatedCount;
  }

  // Pack the updated neighbor count and original row index, then reduce all candidates in this block to one maximum.
  // Doing this here avoids reading the entire neighbor count array again in a separate kernel.
  __shared__ typename cub::BlockReduce<std::uint64_t, kIterationBlockSize>::TempStorage storage;
  const std::uint64_t                                                                   candidate =
    cub::BlockReduce<std::uint64_t, kIterationBlockSize>(storage).Reduce(makeCandidate(updatedCount, row), cubMax());
  if (threadIdx.x == 0) {
    blockMaxima[blockIdx.x] = candidate;
  }
}

// Reduce the blockMaxima array into the centroid
__global__ void selectNextCentroidKernel(const cuda::std::span<const std::uint64_t> blockMaxima,
                                         int*                                       maxValue,
                                         int*                                       maxIndex,
                                         cudaGraphConditionalHandle                 handle) {
  std::uint64_t candidate = 0;
  // Once again we are tiling here
  for (int block = threadIdx.x; block < static_cast<int>(blockMaxima.size()); block += blockDim.x) {
    candidate = max(candidate, blockMaxima[block]);
  }

  __shared__ typename cub::BlockReduce<std::uint64_t, kIterationBlockSize>::TempStorage storage;
  candidate = cub::BlockReduce<std::uint64_t, kIterationBlockSize>(storage).Reduce(candidate, cubMax());
  if (threadIdx.x == 0) {
    // Save the next centroid and continue the CUDA graph while a molecule has an active neighbor besides itself. We do this here instead of in another conditional checking kernel since we already have the maxNeighborCount in a register
    const int maxNeighborCount = storeCandidate(candidate, maxValue, maxIndex);
    cudaGraphSetConditional(handle, maxNeighborCount > 1 ? 1 : 0);
  }
}

// Append all molecules left active after the graph loop as singleton clusters.
__global__ void appendSingletonsKernel(const cuda::std::span<std::uint8_t> active,
                                       const cuda::std::span<int>          clusterMembers,
                                       int*                                outputCount,
                                       int*                                clusterCount,
                                       const cuda::std::span<int>          clusterOffsets,
                                       const cuda::std::span<int>          centroids) {
  // Use one thread so singleton clusters are emitted in deterministic original index order.
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int output  = *outputCount;
  int cluster = *clusterCount;
  // Every remaining active molecule has no active neighbor other than itself, so append it as one complete cluster.
  for (int row = 0; row < static_cast<int>(active.size()); ++row) {
    if (!active[row]) {
      continue;
    }
    active[row]               = 0;
    clusterMembers[output++]  = row;
    centroids[cluster]        = row;
    clusterOffsets[++cluster] = output;
  }
  // Store the final output sizes for the host result copy.
  *outputCount  = output;
  *clusterCount = cluster;
}

// Reordering fused Butina pipeline:
//
// 1) Compute each fingerprint popcount and its initial neighbor count. Similarities are consumed immediately; no
//    N x N similarity, distance, or hit matrix is materialized.
// 2) Find the last active row with the largest neighbor count. This is the first centroid used by the graph.
// 3) Run a conditional CUDA graph WHILE loop entirely on the GPU:
//      a) compare active rows with the selected centroid and append matching rows to the cluster output;
//      b) subtract only those newly removed rows from the surviving neighbor counts;
//      c) reduce a maximum per update block, then reduce those block maxima to select the next centroid;
//      d) update the graph condition from the new maximum count.
//    Fixed original row indices and an active byte per row replace the Python boolean-index compactions.
// 4) Once no centroid has another active neighbor, append all remaining active rows as singleton clusters and copy the
//    completed member, offset, and centroid arrays to the host once.
template <FingerprintSimilarityMetric Metric> class FusedButinaLoopGraph {
 public:
  FusedButinaLoopGraph(const cuda::std::span<const std::uint32_t> fingerprints,
                       const cuda::std::span<const int>           bitCounts,
                       const cuda::std::span<std::uint8_t>        active,
                       const cuda::std::span<int>                 neighborCounts,
                       const cuda::std::span<int>                 clusterMembers,
                       int*                                       clusterStart,
                       int*                                       outputCount,
                       int*                                       clusterCount,
                       const cuda::std::span<int>                 clusterOffsets,
                       const cuda::std::span<int>                 centroids,
                       const cuda::std::span<std::uint64_t>       blockMaxima,
                       int*                                       maxValue,
                       int*                                       maxIndex,
                       int                                        numWords,
                       float                                      threshold) {
    const int numIterationBlocks = static_cast<int>(blockMaxima.size());
    // The conditional handle starts true to provide do-while semantics. If the initial maximum is already a singleton,
    // the body performs no extraction and its final condition immediately exits.
    cudaCheckError(cudaGraphCreate(&graph_, 0));
    cudaCheckError(cudaGraphConditionalHandleCreate(&handle_, graph_, 1, cudaGraphCondAssignDefault));

    cudaGraphNodeParams params = {};
    params.type                = cudaGraphNodeTypeConditional;
    params.conditional.handle  = handle_;
    params.conditional.type    = cudaGraphCondTypeWhile;
    params.conditional.size    = 1;
    cudaGraphNode_t conditionalNode;
#if CUDART_VERSION >= 13000
    cudaCheckError(cudaGraphAddNode(&conditionalNode, graph_, nullptr, nullptr, 0, &params));
#else
    cudaCheckError(cudaGraphAddNode(&conditionalNode, graph_, nullptr, 0, &params));
#endif

    cudaGraph_t  body = params.conditional.phGraph_out[0];
    cudaStream_t captureStream;
    cudaCheckError(cudaStreamCreate(&captureStream));
    cudaCheckError(
      cudaStreamBeginCaptureToGraph(captureStream, body, nullptr, nullptr, 0, cudaStreamCaptureModeRelaxed));

    cudaCheckError(cudaMemcpyAsync(clusterStart, outputCount, sizeof(int), cudaMemcpyDeviceToDevice, captureStream));
    extractClusterKernel<Metric><<<numIterationBlocks, kIterationBlockSize, 0, captureStream>>>(fingerprints,
                                                                                                bitCounts,
                                                                                                active,
                                                                                                clusterMembers,
                                                                                                outputCount,
                                                                                                maxValue,
                                                                                                maxIndex,
                                                                                                numWords,
                                                                                                threshold);
    cudaCheckError(cudaGetLastError());
    recordClusterKernel<<<1, 1, 0, captureStream>>>(maxValue,
                                                    maxIndex,
                                                    outputCount,
                                                    clusterCount,
                                                    clusterOffsets,
                                                    centroids);
    cudaCheckError(cudaGetLastError());
    updateNeighborCountsAndBlockMaxKernel<Metric>
      <<<numIterationBlocks, kIterationBlockSize, 0, captureStream>>>(fingerprints,
                                                                      bitCounts,
                                                                      active,
                                                                      neighborCounts,
                                                                      clusterMembers,
                                                                      clusterStart,
                                                                      outputCount,
                                                                      maxIndex,
                                                                      blockMaxima,
                                                                      numWords,
                                                                      threshold);
    cudaCheckError(cudaGetLastError());
    selectNextCentroidKernel<<<1, kIterationBlockSize, 0, captureStream>>>(blockMaxima, maxValue, maxIndex, handle_);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaStreamEndCapture(captureStream, nullptr));
    cudaCheckError(cudaStreamDestroy(captureStream));

    // Instantiate once per clustering call. All later rounds execute inside this graph with no Python or host loop. =
    cudaCheckError(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0));
  }

  ~FusedButinaLoopGraph() {
    if (graphExec_) {
      cudaGraphExecDestroy(graphExec_);
    }
    if (graph_) {
      cudaGraphDestroy(graph_);
    }
  }

  FusedButinaLoopGraph(const FusedButinaLoopGraph&)            = delete;
  FusedButinaLoopGraph& operator=(const FusedButinaLoopGraph&) = delete;

  void launch(cudaStream_t stream) { cudaCheckError(cudaGraphLaunch(graphExec_, stream)); }

 private:
  cudaGraph_t                graph_     = nullptr;
  cudaGraphExec_t            graphExec_ = nullptr;
  cudaGraphConditionalHandle handle_    = {};
};

template <FingerprintSimilarityMetric Metric>
FusedButinaResult fusedButinaGpuImpl(cuda::std::span<const std::uint32_t> fingerprints,
                                     int                                  n,
                                     int                                  numWords,
                                     float                                threshold,
                                     cudaStream_t                         stream) {
  ScopedNvtxRange setupRange("Fused Butina Setup");

  // All (non scalar) buffers are O(N). Fingerprints remain in caller-owned storage and no pairwise matrix is allocated.
  const int                        numIterationBlocks = (n + kIterationBlockSize - 1) / kIterationBlockSize;
  AsyncDeviceVector<int>           bitCounts(n, stream);
  AsyncDeviceVector<std::uint8_t>  active(n, stream);
  AsyncDeviceVector<int>           neighborCounts(n, stream);
  AsyncDeviceVector<int>           clusterMembers(n, stream);
  AsyncDeviceVector<int>           clusterOffsets(n + 1, stream);
  AsyncDeviceVector<int>           centroids(n, stream);
  AsyncDeviceVector<std::uint64_t> blockMaxima(numIterationBlocks, stream);
  AsyncDevicePtr<int>              clusterStart(0, stream);
  AsyncDevicePtr<int>              outputCount(0, stream);
  AsyncDevicePtr<int>              clusterCount(0, stream);
  AsyncDevicePtr<int>              maxValue(-1, stream);
  AsyncDevicePtr<int>              maxIndex(-1, stream);

  const auto bitCountsSpan      = toSpan(bitCounts);
  const auto activeSpan         = toSpan(active);
  const auto neighborCountsSpan = toSpan(neighborCounts);
  const auto clusterMembersSpan = toSpan(clusterMembers);
  const auto clusterOffsetsSpan = toSpan(clusterOffsets);
  const auto centroidsSpan      = toSpan(centroids);
  const auto blockMaximaSpan    = toSpan(blockMaxima);

  cudaCheckError(cudaMemsetAsync(activeSpan.data(), 1, activeSpan.size_bytes(), stream));
  clusterOffsets.zero();

  // Setup runs once before the graph: cache bit counts, compute initial degrees, and choose the first centroid.
  fingerprintBitCountKernel<<<(n + kIterationBlockSize - 1) / kIterationBlockSize, kIterationBlockSize, 0, stream>>>(
    fingerprints,
    bitCountsSpan,
    numWords);
  cudaCheckError(cudaGetLastError());
  initialNeighborCountKernel<Metric>
    <<<n, kCandidateTileSize, 0, stream>>>(fingerprints, bitCountsSpan, neighborCountsSpan, numWords, threshold);
  cudaCheckError(cudaGetLastError());
  initialArgMaxKernel<<<1, kInitialArgMaxBlockSize, 0, stream>>>(neighborCountsSpan, maxValue.data(), maxIndex.data());
  cudaCheckError(cudaGetLastError());
  setupRange.pop();

  ScopedNvtxRange              buildRange("Build fused Butina graph");
  FusedButinaLoopGraph<Metric> graph(fingerprints,
                                     bitCountsSpan,
                                     activeSpan,
                                     neighborCountsSpan,
                                     clusterMembersSpan,
                                     clusterStart.data(),
                                     outputCount.data(),
                                     clusterCount.data(),
                                     clusterOffsetsSpan,
                                     centroidsSpan,
                                     blockMaximaSpan,
                                     maxValue.data(),
                                     maxIndex.data(),
                                     numWords,
                                     threshold);
  buildRange.pop();

  const ScopedNvtxRange loopRange("Fused Butina graph loop");
  graph.launch(stream);

  // Rows left after the graph have no active non-self neighbor and can be emitted as singleton clusters directly.
  appendSingletonsKernel<<<1, 1, 0, stream>>>(activeSpan,
                                              clusterMembersSpan,
                                              outputCount.data(),
                                              clusterCount.data(),
                                              clusterOffsetsSpan,
                                              centroidsSpan);
  cudaCheckError(cudaGetLastError());

  // Copy the completed result to the host with one synchronization.
  int              outputCountHost  = 0;
  int              clusterCountHost = 0;
  std::vector<int> membersHost(n);
  std::vector<int> offsetsHost(n + 1);
  std::vector<int> centroidsHost(n);
  outputCount.get(outputCountHost);
  clusterCount.get(clusterCountHost);
  clusterMembers.copyToHost(membersHost);
  clusterOffsets.copyToHost(offsetsHost);
  centroids.copyToHost(centroidsHost);
  cudaCheckError(cudaStreamSynchronize(stream));

  if (outputCountHost != n || clusterCountHost < 0 || clusterCountHost > n) {
    throw std::runtime_error("Fused Butina produced inconsistent output sizes");
  }
  membersHost.resize(n);
  offsetsHost.resize(clusterCountHost + 1);
  centroidsHost.resize(clusterCountHost);
  return {std::move(membersHost), std::move(offsetsHost), std::move(centroidsHost)};
}

}  // namespace

FusedButinaResult fusedButinaGpu(cuda::std::span<const std::uint32_t> fingerprints,
                                 int                                  numFingerprints,
                                 int                                  numWords,
                                 double                               cutoff,
                                 FingerprintSimilarityMetric          metric,
                                 cudaStream_t                         stream) {
  // Validate dimensions before allocating device buffers. The Python wrapper performs dtype and contiguity checks.
  if (numFingerprints < 0 || numWords <= 0) {
    throw std::invalid_argument("Fingerprint matrix dimensions must be non-negative with at least one word");
  }
  if (fingerprints.size() != static_cast<std::size_t>(numFingerprints) * static_cast<std::size_t>(numWords)) {
    throw std::invalid_argument("Fingerprint buffer size does not match its shape");
  }
  if (cutoff < 0.0 || cutoff > 1.0) {
    throw std::invalid_argument("cutoff must be in [0, 1]");
  }
  if (metric != FingerprintSimilarityMetric::Tanimoto && metric != FingerprintSimilarityMetric::Cosine) {
    throw std::invalid_argument("Unsupported fingerprint similarity metric");
  }
  if (numFingerprints == 0) {
    return {{}, {0}, {}};
  }

  // Compile separate metric-specialized graphs so the hot pair predicate contains no runtime metric branch.
  const float threshold = static_cast<float>(1.0 - cutoff);
  if (metric == FingerprintSimilarityMetric::Tanimoto) {
    return fusedButinaGpuImpl<FingerprintSimilarityMetric::Tanimoto>(fingerprints,
                                                                     numFingerprints,
                                                                     numWords,
                                                                     threshold,
                                                                     stream);
  }
  return fusedButinaGpuImpl<FingerprintSimilarityMetric::Cosine>(fingerprints,
                                                                 numFingerprints,
                                                                 numWords,
                                                                 threshold,
                                                                 stream);
}

}  // namespace nvMolKit
