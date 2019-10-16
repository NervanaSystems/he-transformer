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
#include "seal/kernel/exp_seal.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {

void scalar_exp_seal(const HEPlaintext& arg, HEPlaintext& out) {
  const std::vector<double>& arg_vals = arg.values();
  std::vector<double> out_vals(arg.num_values());

  auto exp = [](double d) { return std::exp(d); };
  std::transform(arg_vals.begin(), arg_vals.end(), out_vals.begin(), exp);
  out.set_values(out_vals);
}

void exp_seal(const std::vector<HEPlaintext>& arg,
              std::vector<HEPlaintext>& out, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    scalar_exp_seal(arg[i], out[i]);
  }
}

void scalar_exp_seal_known_value(const SealCiphertextWrapper& arg,
                                 std::shared_ptr<SealCiphertextWrapper>& out) {
  auto exp = [](double d) { return std::exp(d); };
  NGRAPH_CHECK(arg.known_value());
  out->known_value() = true;
  out->value() = exp(arg.value());
}

void scalar_exp_seal(const SealCiphertextWrapper& arg,
                     std::shared_ptr<SealCiphertextWrapper>& out,
                     const seal::parms_id_type& parms_id, double scale,
                     seal::CKKSEncoder& ckks_encoder,
                     seal::Encryptor& encryptor, seal::Decryptor& decryptor) {
  auto exp = [](double d) { return std::exp(d); };

  if (arg.known_value()) {
    scalar_exp_seal_known_value(arg, out);
  } else {
    HEPlaintext plain;
    decrypt(plain, arg, decryptor, ckks_encoder);
    const std::vector<double>& arg_vals = plain.values();
    std::vector<double> out_vals(plain.num_values());
    std::transform(arg_vals.begin(), arg_vals.end(), out_vals.begin(), exp);
    plain.set_values(out_vals);
    encrypt(out, plain, parms_id, ngraph::element::f32, scale, ckks_encoder,
            encryptor, arg.complex_packing());
  }
}

void scalar_exp_seal(const SealCiphertextWrapper& arg,
                     std::shared_ptr<SealCiphertextWrapper>& out,
                     const HESealBackend& he_seal_backend) {
  scalar_exp_seal(
      arg, out, he_seal_backend.get_context()->first_parms_id(),
      he_seal_backend.get_scale(), *he_seal_backend.get_ckks_encoder(),
      *he_seal_backend.get_encryptor(), *he_seal_backend.get_decryptor());
}

void exp_seal(const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg,
              std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
              size_t count, const HESealBackend& he_seal_backend) {
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_exp_seal(*arg[i], out[i], he_seal_backend);
  }
}

}  // namespace he
}  // namespace ngraph