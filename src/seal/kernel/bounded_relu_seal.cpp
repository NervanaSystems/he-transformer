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

#include <memory>
#include <vector>

#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/bounded_relu_seal.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {

void scalar_bounded_relu_seal(const HEPlaintext& arg, HEPlaintext& out,
                              float alpha) {
  std::vector<double> out_vals(arg.size());

  auto bounded_relu = [alpha](double f) {
    return f > alpha ? alpha : (f > 0) ? f : 0.f;
  };
  std::transform(arg.begin(), arg.end(), out_vals.begin(), bounded_relu);
  out = HEPlaintext(std::vector<double>{out_vals});
}

void scalar_bounded_relu_seal(const HEType& arg, HEType& out, float alpha,
                              const seal::parms_id_type& parms_id, double scale,
                              seal::CKKSEncoder& ckks_encoder,
                              seal::Encryptor& encryptor,
                              seal::Decryptor& decryptor) {
  if (arg.is_plaintext()) {
    out.set_plaintext(arg.get_plaintext());
    scalar_bounded_relu_seal(arg.get_plaintext(), out.get_plaintext(), alpha);
  } else {
    HEPlaintext plain;
    decrypt(plain, *arg.get_ciphertext(), arg.complex_packing(), decryptor,
            ckks_encoder);
    scalar_bounded_relu_seal(plain, plain, alpha);
    encrypt(out.get_ciphertext(), plain, parms_id, ngraph::element::f32, scale,
            ckks_encoder, encryptor, arg.complex_packing());
  }
}

void scalar_bounded_relu_seal(const HEType& arg, HEType& out, float alpha,
                              const HESealBackend& he_seal_backend) {
  scalar_bounded_relu_seal(
      arg, out, alpha, he_seal_backend.get_context()->first_parms_id(),
      he_seal_backend.get_scale(), *he_seal_backend.get_ckks_encoder(),
      *he_seal_backend.get_encryptor(), *he_seal_backend.get_decryptor());
}

void bounded_relu_seal(const std::vector<HEType>& arg, std::vector<HEType>& out,
                       float alpha, size_t count,
                       const HESealBackend& he_seal_backend) {
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_bounded_relu_seal(arg[i], out[i], alpha, he_seal_backend);
  }
}

}  // namespace he
}  // namespace ngraph
