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

#include "seal/kernel/subtract_seal.hpp"
#include "seal/kernel/add_seal.hpp"
#include "seal/kernel/negate_seal.hpp"

void ngraph::he::scalar_subtract_seal(
    ngraph::he::SealCiphertextWrapper& arg0,
    ngraph::he::SealCiphertextWrapper& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend) {
  if (arg0.is_zero()) {
    he_seal_backend->get_evaluator()->negate(arg1.ciphertext(),
                                             out->ciphertext());
  } else if (arg1.is_zero()) {
    out = std::make_shared<ngraph::he::SealCiphertextWrapper>(arg0);
  } else {
    he_seal_backend->get_evaluator()->sub(arg0.ciphertext(), arg1.ciphertext(),
                                          out->ciphertext());
  }
}

void ngraph::he::scalar_subtract_seal(
    ngraph::he::SealCiphertextWrapper& arg0, const HEPlaintext& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend) {
  auto p = SealPlaintextWrapper(arg0.complex_packing());
  he_seal_backend->encode(p, arg1, arg0.ciphertext().parms_id(),
                          arg0.ciphertext().scale());
  he_seal_backend->get_evaluator()->sub_plain(arg0.ciphertext(), p.plaintext(),
                                              out->ciphertext());
}

void ngraph::he::scalar_subtract_seal(
    const HEPlaintext& arg0, SealCiphertextWrapper& arg1,
    std::shared_ptr<SealCiphertextWrapper>& out, const element::Type& type,
    const ngraph::he::HESealBackend* he_seal_backend) {
  if (arg1.is_zero()) {
    he_seal_backend->encrypt(out, arg0);
  } else {
    auto tmp = std::make_shared<ngraph::he::SealCiphertextWrapper>();
    ngraph::he::scalar_negate_seal(arg1, tmp, type, he_seal_backend);
    ngraph::he::scalar_add_seal(arg0, *tmp, out, type, he_seal_backend);
  }
}

void ngraph::he::scalar_subtract_seal(const HEPlaintext& arg0,
                                      const HEPlaintext& arg1, HEPlaintext& out,
                                      const element::Type& element_type,
                                      const HESealBackend* he_seal_backend) {
  NGRAPH_CHECK(element_type == element::f32);

  const std::vector<float>& arg0_vals = arg0.get_values();
  const std::vector<float>& arg1_vals = arg1.get_values();
  std::vector<float> out_vals(arg0.num_values());

  std::transform(arg0_vals.begin(), arg0_vals.end(), arg1_vals.begin(),
                 out_vals.begin(), std::minus<float>());
  out.set_values(out_vals);
}
