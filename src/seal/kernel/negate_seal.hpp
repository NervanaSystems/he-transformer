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

#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace he {
void scalar_negate_seal(const SealCiphertextWrapper& arg,
                        std::shared_ptr<SealCiphertextWrapper>& out,
                        const element::Type& element_type,
                        const HESealBackend& he_seal_backend);

void scalar_negate_seal(const HEPlaintext& arg, HEPlaintext& out,
                        const element::Type& element_type);

inline void negate_seal(const std::vector<HEPlaintext>& arg,
                        std::vector<HEPlaintext>& out,
                        const element::Type& element_type, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    scalar_negate_seal(arg[i], out[i], element_type);
  }
}

inline void negate_seal(
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const element::Type& element_type,
    const ngraph::he::HESealBackend& he_seal_backend, size_t count) {
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_negate_seal(*arg[i], out[i], element_type, he_seal_backend);
  }
}
}  // namespace he
}  // namespace ngraph
