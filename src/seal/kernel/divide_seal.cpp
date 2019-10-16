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
#include "seal/kernel/add_seal.hpp"
#include "seal/kernel/negate_seal.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {

void scalar_divide_seal(SealCiphertextWrapper& arg0,
                        SealCiphertextWrapper& arg1,
                        std::shared_ptr<SealCiphertextWrapper>& out,
                        const element::Type& element_type,
                        HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
  if (arg0.known_value() && arg1.known_value()) {
    out->known_value() = true;
    out->value() = arg0.value() / arg1.value();
  } else if (arg0.known_value()) {
    HEPlaintext p(arg0.value());
    scalar_divide_seal(p, arg1, out, element_type, he_seal_backend);
    out->known_value() = false;
  } else if (arg1.known_value()) {
    HEPlaintext p(arg1.value());
    scalar_divide_seal(arg0, p, out, element_type, he_seal_backend);
    out->known_value() = false;
  } else {
    he_seal_backend.get_evaluator()->sub(arg0.ciphertext(), arg1.ciphertext(),
                                         out->ciphertext());
    out->known_value() = false;
  }
}

void scalar_divide_seal(SealCiphertextWrapper& arg0, const HEPlaintext& arg1,
                        std::shared_ptr<SealCiphertextWrapper>& out,
                        const element::Type& element_type,
                        HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
  if (arg0.known_value()) {
    NGRAPH_CHECK(arg1.is_single_value(), "arg1 is not single value");
    out->known_value() = true;
    out->value() = arg0.value() / arg1.first_value();
    out->complex_packing() = arg0.complex_packing();
  } else {
    auto p = SealPlaintextWrapper(arg0.complex_packing());
    encode(p, arg1, *he_seal_backend.get_ckks_encoder(),
           arg0.ciphertext().parms_id(), element_type,
           arg0.ciphertext().scale(), arg0.complex_packing());
    he_seal_backend.get_evaluator()->sub_plain(arg0.ciphertext(), p.plaintext(),
                                               out->ciphertext());
    out->known_value() = false;
  }
}

void scalar_divide_seal(const HEPlaintext& arg0, SealCiphertextWrapper& arg1,
                        std::shared_ptr<SealCiphertextWrapper>& out,
                        const element::Type& element_type,
                        HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
  if (arg1.known_value()) {
    NGRAPH_CHECK(arg0.is_single_value(), "arg0 is not single value");
    out->known_value() = true;
    out->value() = arg0.first_value() / arg1.value();
    out->complex_packing() = arg1.complex_packing();
  } else {
    auto tmp = std::make_shared<SealCiphertextWrapper>();
    scalar_negate_seal(arg1, tmp, element_type, he_seal_backend);
    scalar_add_seal(arg0, *tmp, out, element_type, he_seal_backend);
    out->known_value() = false;
  }
}

void scalar_divide_seal(const HEPlaintext& arg0, const HEPlaintext& arg1,
                        HEPlaintext& out) {
  const std::vector<double>& arg0_vals = arg0.values();
  const std::vector<double>& arg1_vals = arg1.values();
  std::vector<double> out_vals(arg0.num_values());

  std::transform(arg0_vals.begin(), arg0_vals.end(), arg1_vals.begin(),
                 out_vals.begin(), std::divides<double>());
  out.set_values(out_vals);
}

}  // namespace he
}  // namespace ngraph