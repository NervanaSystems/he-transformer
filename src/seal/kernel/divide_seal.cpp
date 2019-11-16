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

#include "seal/kernel/divide_seal.hpp"

#include <memory>
#include <vector>

#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/multiply_seal.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"

namespace ngraph::runtime::he {

void scalar_divide_seal(const HEPlaintext& arg0, const HEPlaintext& arg1,
                        HEPlaintext& out) {
  out.resize(arg0.size());
  std::transform(arg0.begin(), arg0.end(), arg1.begin(), out.begin(),
                 std::divides<>());
}

void scalar_divide_seal(HEType& arg0, HEType& arg1, HEType& out,
                        HESealBackend& he_seal_backend) {
  if (arg0.is_ciphertext() && arg1.is_ciphertext()) {
    NGRAPH_CHECK(arg0.complex_packing() == arg1.complex_packing(),
                 "Complex packing types don't match");
    NGRAPH_WARN << " Dividing ciphertext / ciphertext without client "
                   "is not privacy-preserving ";

    // TODO(fboemer): enable with client?
    // TODO(fboemer): complex packing?
    HEPlaintext plain_arg0;
    HEPlaintext plain_arg1;
    he_seal_backend.decrypt(plain_arg0, *arg0.get_ciphertext(),
                            arg0.batch_size(), arg0.complex_packing());
    he_seal_backend.decrypt(plain_arg1, *arg1.get_ciphertext(),
                            arg1.batch_size(), arg1.complex_packing());
    scalar_divide_seal(plain_arg0, plain_arg1, plain_arg1);

    he_seal_backend.encrypt(out.get_ciphertext(), plain_arg1, element::f32,
                            arg0.complex_packing());

  } else if (arg0.is_ciphertext() && arg1.is_plaintext()) {
    HEType arg1_inv = arg1;
    HEPlaintext& arg1_plain = arg1.get_plaintext();
    HEPlaintext& arg1_inv_plain = arg1_inv.get_plaintext();
    for (size_t i = 0; i < arg1.get_plaintext().size(); ++i) {
      arg1_inv_plain[i] = 1 / arg1_plain[i];
    }
    scalar_multiply_seal(arg0, arg1_inv, out, he_seal_backend);

  } else if (arg0.is_plaintext() && arg1.is_ciphertext()) {
    NGRAPH_WARN << " Dividing plaintext / ciphertext without client "
                   "is not privacy-preserving ";

    // TODO(fboemer): enable with client?
    // TODO(fboemer): complex packing?
    HEPlaintext plain_arg1;
    he_seal_backend.decrypt(plain_arg1, *arg1.get_ciphertext(),
                            arg1.batch_size(), arg1.complex_packing());
    scalar_divide_seal(arg0.get_plaintext(), plain_arg1, plain_arg1);
    he_seal_backend.encrypt(out.get_ciphertext(), plain_arg1, element::f32,
                            arg0.complex_packing());

  } else if (arg0.is_plaintext() && arg1.is_plaintext()) {
    out.set_plaintext(arg0.get_plaintext());
    scalar_divide_seal(arg0.get_plaintext(), arg1.get_plaintext(),
                       out.get_plaintext());
  }
}

void divide_seal(std::vector<HEType>& arg0, std::vector<HEType>& arg1,
                 std::vector<HEType>& out, size_t count,
                 const element::Type& element_type,
                 HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);

#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_divide_seal(arg0[i], arg1[i], out[i], he_seal_backend);
  }
}

}  // namespace ngraph::runtime::he
