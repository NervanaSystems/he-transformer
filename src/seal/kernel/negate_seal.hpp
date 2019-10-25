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
                        const HESealBackend& he_seal_backend);

void scalar_negate_seal(const HEPlaintext& arg, HEPlaintext& out);

inline void scalar_negate_seal(HEType& arg, HEType& out,
                               const element::Type& element_type,
                               const HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
  NGRAPH_CHECK(arg.complex_packing() == out.complex_packing(),
               "Complex packing types don't match");
  out.complex_packing() = arg.complex_packing();

  if (arg.is_ciphertext() && out.is_ciphertext()) {
    scalar_negate_seal(*arg.get_ciphertext(), out.get_ciphertext(),
                       he_seal_backend);
  } else if (arg.is_plaintext() && out.is_plaintext()) {
    scalar_negate_seal(arg.get_plaintext(), out.get_plaintext());
  } else {
    NGRAPH_CHECK(false, "Unknown argument types");
  }
}

inline void negate_seal(std::vector<HEType>& arg, std::vector<HEType>& out,
                        size_t count, const element::Type& element_type,
                        const HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
  NGRAPH_CHECK(count <= arg.size(), "Count ", count,
               " is too large for arg, with size ", arg.size());
  NGRAPH_CHECK(count <= out.size(), "Count ", count,
               " is too large for out, with size ", out.size());

#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_negate_seal(arg[i], out[i], element_type, he_seal_backend);
  }
}

}  // namespace he
}  // namespace ngraph
