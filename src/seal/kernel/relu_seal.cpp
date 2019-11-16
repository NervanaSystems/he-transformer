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

#include "seal/kernel/relu_seal.hpp"

#include <memory>
#include <vector>

#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"

namespace ngraph::runtime::he {

void scalar_relu_seal(const HEPlaintext& arg, HEPlaintext& out) {
  out.resize(arg.size());
  auto relu = [](double d) { return d > 0 ? d : 0.; };
  std::transform(arg.begin(), arg.end(), out.begin(), relu);
}

void scalar_relu_seal(const HEType& arg, HEType& out,
                      const seal::parms_id_type& parms_id, double scale,
                      seal::CKKSEncoder& ckks_encoder,
                      seal::Encryptor& encryptor, seal::Decryptor& decryptor,
                      std::shared_ptr<seal::SEALContext> context) {
  if (arg.is_plaintext()) {
    out.set_plaintext(arg.get_plaintext());
    scalar_relu_seal(arg.get_plaintext(), out.get_plaintext());
  } else {
    HEPlaintext plain;
    decrypt(plain, *arg.get_ciphertext(), arg.complex_packing(), decryptor,
            ckks_encoder, context, arg.batch_size());
    scalar_relu_seal(plain, plain);
    encrypt(out.get_ciphertext(), plain, parms_id, element::f32, scale,
            ckks_encoder, encryptor, arg.complex_packing());
    out.set_ciphertext(out.get_ciphertext());
  }
}

void scalar_relu_seal(const HEType& arg, HEType& out,
                      const HESealBackend& he_seal_backend) {
  scalar_relu_seal(
      arg, out, he_seal_backend.get_context()->first_parms_id(),
      he_seal_backend.get_scale(), *he_seal_backend.get_ckks_encoder(),
      *he_seal_backend.get_encryptor(), *he_seal_backend.get_decryptor(),
      he_seal_backend.get_context());
}

void relu_seal(const std::vector<HEType>& arg, std::vector<HEType>& out,
               size_t count, const HESealBackend& he_seal_backend) {
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_relu_seal(arg[i], out[i], he_seal_backend);
  }
}

}  // namespace ngraph::runtime::he
