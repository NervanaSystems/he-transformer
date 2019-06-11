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

#pragma once

#include <memory>
#include <vector>

#include "he_plaintext.hpp"
#include "ngraph/check.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {
inline void result_seal(
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out, size_t count) {
  NGRAPH_CHECK(out.size() == arg.size(), "Result output size ", out.size(),
               " does not match result input size ", arg.size());
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    out[i] = arg[i];
  }
}

inline void result_seal(const std::vector<HEPlaintext>& arg,
                        std::vector<HEPlaintext>& out, size_t count) {
  NGRAPH_CHECK(out.size() == arg.size(), "Result output size ", out.size(),
               " does not match result input size ", arg.size());
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    out[i] = arg[i];
  }
}

void result_seal(const std::vector<HEPlaintext>& arg,
                 std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
                 size_t count, const HESealBackend& he_seal_backend);

void result_seal(const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg,
                 std::vector<HEPlaintext>& out, size_t count,
                 const HESealBackend& he_seal_backend);
}  // namespace he
}  // namespace ngraph
