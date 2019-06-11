//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "seal/kernel/result_seal.hpp"

void ngraph::he::result_seal(
    const std::vector<HEPlaintext>& arg,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out, size_t count,
    const HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(out.size() == arg.size(), "Result output size ", out.size(),
               " does not match result input size ", arg.size());
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    he_seal_backend.encrypt(out[i], arg[i], he_seal_backend.complex_packing());
  }
}

void ngraph::he::result_seal(
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg,
    std::vector<HEPlaintext>& out, size_t count,
    const HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(out.size() == arg.size(), "Result output size ", out.size(),
               " does not match result input size ", arg.size());
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    he_seal_backend.decrypt(out[i], *arg[i]);
  }
}
