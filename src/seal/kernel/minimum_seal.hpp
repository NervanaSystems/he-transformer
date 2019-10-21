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

#include "he_plaintext.hpp"
#include "he_type.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph {
namespace he {

void scalar_minimum_seal(SealCiphertextWrapper& arg0,
                         SealCiphertextWrapper& arg1,
                         std::shared_ptr<SealCiphertextWrapper>& out,
                         const bool complex_packing,
                         HESealBackend& he_seal_backend);

void scalar_minimum_seal(SealCiphertextWrapper& arg0, const HEPlaintext& arg1,
                         HEType& out, HESealBackend& he_seal_backend);

inline void scalar_minimum_seal(const HEPlaintext& arg0,
                                SealCiphertextWrapper& arg1, HEType& out,
                                HESealBackend& he_seal_backend) {
  scalar_minimum_seal(arg1, arg0, out, he_seal_backend, pool);
}

void scalar_minimum_seal(const HEPlaintext& arg0, const HEPlaintext& arg1,
                         HEPlaintext& out);

inline void scalar_minimum_seal(HEType& arg0, HEType& arg1, HEType& out,
                                HESealBackend& he_seal_backend) {
  if (arg0.is_ciphertext() && arg1.is_ciphertext()) {
    NGRAPH_CHECK(arg0.complex_packing() == arg1.complex_packing(),
                 "Complex packing types don't match");
    if (!out.is_ciphertext()) {
      out.set_ciphertext(HESealBackend::create_empty_ciphertext());
    }

    scalar_minimum_seal(*arg0.get_ciphertext(), *arg1.get_ciphertext(),
                        out.get_ciphertext(), arg0.complex_packing(),
                        he_seal_backend);
  } else if (arg0.is_ciphertext() && arg1.is_plaintext()) {
    if (!out.is_ciphertext()) {
      out.set_ciphertext(HESealBackend::create_empty_ciphertext());
    }
    scalar_minimum_seal(*arg0.get_ciphertext(), arg1.get_plaintext(), out,
                        he_seal_backend);
  } else if (arg0.is_plaintext() && arg1.is_ciphertext()) {
    if (!out.is_ciphertext()) {
      out.set_ciphertext(HESealBackend::create_empty_ciphertext());
    }
    scalar_minimum_seal(*arg1.get_ciphertext(), arg0.get_plaintext(), out,
                        he_seal_backend);
  } else if (arg0.is_plaintext() && arg1.is_plaintext()) {
    NGRAPH_CHECK(arg0.complex_packing() == arg1.complex_packing(),
                 "Complex packing types don't match");
    if (!out.is_plaintext()) {
      out.set_plaintext(HEPlaintext());
    }

    scalar_minimum_seal(arg0.get_plaintext(), arg1.get_plaintext(),
                        out.get_plaintext());
  } else {
    NGRAPH_CHECK(false, "Unknown argument types");
  }
  out.complex_packing() = arg0.complex_packing();
}

inline void minimum_seal(const std::vector<HEType>& arg0,
                         const std::vector<HEType>& arg1,
                         std::vector<HEType>& out, size_t count,
                         HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(arg0.size() == arg1.size(), "arg0.size() = ", arg0.size(),
               " does not match arg1.size()", arg1.size());
  NGRAPH_CHECK(arg0.size() == out.size(), "arg0.size() = ", arg0.size(),
               " does not match out.size()", out.size());
  for (size_t i = 0; i < count; ++i) {
    scalar_minimum_seal(arg0[i], arg1[i], out[i], he_seal_backend);
  }
}

}  // namespace he
}  // namespace ngraph
