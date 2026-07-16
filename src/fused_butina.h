// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_FUSED_BUTINA_H
#define NVMOLKIT_FUSED_BUTINA_H

#include <cuda_runtime.h>

#include <cstdint>
#include <cuda/std/span>
#include <vector>

namespace nvMolKit {

enum class FingerprintSimilarityMetric {
  Tanimoto,
  Cosine
};

struct FusedButinaResult {
  std::vector<int> clusterMembers;
  std::vector<int> clusterOffsets;
  std::vector<int> centroids;
};

FusedButinaResult fusedButinaGpu(cuda::std::span<const std::uint32_t> fingerprints,
                                 int                                  numFingerprints,
                                 int                                  numWords,
                                 double                               cutoff,
                                 FingerprintSimilarityMetric          metric,
                                 cudaStream_t                         stream = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_FUSED_BUTINA_H
