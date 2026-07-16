// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <boost/python.hpp>
#include <boost/python/manage_new_object.hpp>
#include <memory>

#include "nvmolkit/array_helpers.h"
#include "src/butina.h"
#include "src/fused_butina.h"
#include "src/utils/device.h"

namespace {

boost::python::object toOwnedPyArray(nvMolKit::PyArray* array) {
  using Converter = boost::python::manage_new_object::apply<nvMolKit::PyArray*>::type;
  return boost::python::object(boost::python::handle<>(Converter()(array)));
}

}  // namespace

BOOST_PYTHON_MODULE(_clustering) {
  boost::python::def(
    "butina",
    +[](const boost::python::dict& distanceMatrix,
        const double               cutoff,
        const int                  neighborlistMaxSize,
        const bool                 returnCentroids,
        const bool                 reordering,
        std::uintptr_t             streamPtr) -> boost::python::object {
      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }
      auto                             stream  = *streamOpt;
      // Extract boost::python::tuple from dict['shape']
      boost::python::tuple             shape   = boost::python::extract<boost::python::tuple>(distanceMatrix["shape"]);
      const size_t                     matDim1 = boost::python::extract<size_t>(shape[0]);
      nvMolKit::AsyncDeviceVector<int> clusterIds(matDim1, stream);
      nvMolKit::AsyncDeviceVector<int> centroids;

      boost::python::tuple data        = boost::python::extract<boost::python::tuple>(distanceMatrix["data"]);
      const size_t         dataPointer = boost::python::extract<std::size_t>(data[0]);
      const auto matSpan = nvMolKit::getSpanFromDictElems<double>(reinterpret_cast<void*>(dataPointer), shape);
      if (returnCentroids) {
        centroids.resize(matDim1);
        centroids.setStream(stream);
      }
      const auto centroidSpan = returnCentroids ? toSpan(centroids) : cuda::std::span<int>{};
      const int  numClusters =
        nvMolKit::butinaGpu(matSpan, toSpan(clusterIds), cutoff, neighborlistMaxSize, centroidSpan, reordering, stream);
      if (returnCentroids) {
        auto clusterArray  = nvMolKit::makePyArray(clusterIds, boost::python::make_tuple(matDim1));
        auto centroidArray = nvMolKit::makePyArray(centroids, boost::python::make_tuple(numClusters));
        return boost::python::make_tuple(toOwnedPyArray(clusterArray), toOwnedPyArray(centroidArray));
      }

      return toOwnedPyArray(nvMolKit::makePyArray(clusterIds, boost::python::make_tuple(matDim1)));
    },
    (boost::python::arg("distance_matrix"),
     boost::python::arg("cutoff"),
     boost::python::arg("neighborlist_max_size") = 64,
     boost::python::arg("return_centroids")      = false,
     boost::python::arg("reordering")            = true,
     boost::python::arg("stream")                = 0));

  boost::python::def(
    "fused_butina",
    +[](const boost::python::dict& fingerprints,
        const double               cutoff,
        const std::string&         metric,
        std::uintptr_t             streamPtr) -> boost::python::tuple {
      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }
      const auto           stream = *streamOpt;
      boost::python::tuple shape  = boost::python::extract<boost::python::tuple>(fingerprints["shape"]);
      if (len(shape) != 2) {
        throw std::invalid_argument("fingerprints must be a 2D matrix");
      }
      const int            n           = boost::python::extract<int>(shape[0]);
      const int            numWords    = boost::python::extract<int>(shape[1]);
      boost::python::tuple data        = boost::python::extract<boost::python::tuple>(fingerprints["data"]);
      const std::size_t    dataPointer = boost::python::extract<std::size_t>(data[0]);
      const auto span = nvMolKit::getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(dataPointer), shape);

      nvMolKit::FingerprintSimilarityMetric parsedMetric;
      if (metric == "tanimoto") {
        parsedMetric = nvMolKit::FingerprintSimilarityMetric::Tanimoto;
      } else if (metric == "cosine") {
        parsedMetric = nvMolKit::FingerprintSimilarityMetric::Cosine;
      } else {
        throw std::invalid_argument("metric must be one of ['tanimoto', 'cosine']");
      }

      auto                result = nvMolKit::fusedButinaGpu(span, n, numWords, cutoff, parsedMetric, stream);
      boost::python::list members;
      boost::python::list offsets;
      boost::python::list centroids;
      for (const int value : result.clusterMembers) {
        members.append(value);
      }
      for (const int value : result.clusterOffsets) {
        offsets.append(value);
      }
      for (const int value : result.centroids) {
        centroids.append(value);
      }
      return boost::python::make_tuple(members, offsets, centroids);
    },
    (boost::python::arg("fingerprints"),
     boost::python::arg("cutoff"),
     boost::python::arg("metric") = "tanimoto",
     boost::python::arg("stream") = 0));
};
