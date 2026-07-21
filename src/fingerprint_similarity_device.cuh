// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_FINGERPRINT_SIMILARITY_DEVICE_CUH
#define NVMOLKIT_FINGERPRINT_SIMILARITY_DEVICE_CUH

#include <cuda_runtime.h>

#include <cmath>
#include <type_traits>

namespace nvMolKit {

enum class FingerprintSimilarityMetric {
  Tanimoto,
  Cosine
};

namespace detail {

template <FingerprintSimilarityMetric Metric, typename Real>
__device__ __forceinline__ Real fingerprintSimilarityFromDenominator(const int intersection, const Real denominator) {
  return denominator > Real{0} ? static_cast<Real>(intersection) / denominator :
                                 (Metric == FingerprintSimilarityMetric::Tanimoto ? Real{1} : Real{0});
}

template <FingerprintSimilarityMetric Metric, typename Real>
__device__ __forceinline__ Real fingerprintSimilarityDenominator(const int intersection,
                                                                 const int lhsBitCount,
                                                                 const int rhsBitCount) {
  if constexpr (Metric == FingerprintSimilarityMetric::Tanimoto) {
    return static_cast<Real>(lhsBitCount + rhsBitCount - intersection);
  } else if constexpr (Metric == FingerprintSimilarityMetric::Cosine) {
    const Real product = static_cast<Real>(lhsBitCount) * static_cast<Real>(rhsBitCount);
    if constexpr (std::is_same_v<Real, float>) {
      return sqrtf(product);
    } else {
      return sqrt(product);
    }
  } else {
    static_assert(Metric == FingerprintSimilarityMetric::Tanimoto || Metric == FingerprintSimilarityMetric::Cosine,
                  "Unsupported fingerprint similarity metric");
  }
}

template <FingerprintSimilarityMetric Metric, typename Real>
__device__ __forceinline__ Real fingerprintSimilarity(const int intersection,
                                                      const int lhsBitCount,
                                                      const int rhsBitCount) {
  return fingerprintSimilarityFromDenominator<Metric>(
    intersection,
    fingerprintSimilarityDenominator<Metric, Real>(intersection, lhsBitCount, rhsBitCount));
}

template <FingerprintSimilarityMetric Metric>
__device__ __forceinline__ bool fingerprintSimilarityAtLeast(const int   intersection,
                                                             const int   lhsBitCount,
                                                             const int   rhsBitCount,
                                                             const float threshold) {
  const float denominator = fingerprintSimilarityDenominator<Metric, float>(intersection, lhsBitCount, rhsBitCount);
  return denominator > 0.0F ? static_cast<float>(intersection) >= threshold * denominator :
                              fingerprintSimilarityFromDenominator<Metric>(intersection, denominator) >= threshold;
}

}  // namespace detail
}  // namespace nvMolKit

#endif  // NVMOLKIT_FINGERPRINT_SIMILARITY_DEVICE_CUH
