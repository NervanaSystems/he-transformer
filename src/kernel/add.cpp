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

#include "kernel/add.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/add_seal.hpp"

void ngraph::he::scalar_add(std::shared_ptr<ngraph::he::HECiphertext>& arg0,
                            std::shared_ptr<ngraph::he::HECiphertext>& arg1,
                            std::shared_ptr<ngraph::he::HECiphertext>& out,
                            const element::Type& element_type,
                            const ngraph::he::HEBackend* he_backend) {
  auto he_seal_backend = ngraph::he::cast_to_seal_backend(he_backend);
  auto arg0_seal = ngraph::he::cast_to_seal_hetext(arg0);
  auto arg1_seal = ngraph::he::cast_to_seal_hetext(arg1);
  auto out_seal = ngraph::he::cast_to_seal_hetext(out);

  ngraph::he::scalar_add(arg0_seal, arg1_seal, out_seal, element_type,
                         he_seal_backend);
  out = std::dynamic_pointer_cast<ngraph::he::HECiphertext>(out_seal);
}

void ngraph::he::scalar_add(const HEPlaintext& arg0, const HEPlaintext& arg1,
                            HEPlaintext& out, const element::Type& element_type,
                            const ngraph::he::HEBackend* he_backend) {
  std::vector<float> arg0_vals = arg0.get_values();
  std::vector<float> arg1_vals = arg1.get_values();
  std::vector<float> out_vals(arg0.num_values());

  std::transform(arg0_vals.begin(), arg0_vals.end(), arg1_vals.begin(),
                 out_vals.begin(), std::plus<float>());

  out.set_values(out_vals);
}

void ngraph::he::scalar_add(std::shared_ptr<HECiphertext>& arg0,
                            const HEPlaintext& arg1,
                            std::shared_ptr<HECiphertext>& out,
                            const element::Type& element_type,
                            const ngraph::he::HEBackend* he_backend) {
  if (arg0->is_zero()) {
    he_backend->encrypt(out, arg1);
    return;
  }
  auto he_seal_backend = cast_to_seal_backend(he_backend);
  auto arg0_seal = ngraph::he::cast_to_seal_hetext(arg0);
  auto out_seal = ngraph::he::cast_to_seal_hetext(out);

  ngraph::he::scalar_add(arg0_seal, arg1, out_seal, element_type,
                         he_seal_backend);
  out = std::dynamic_pointer_cast<HECiphertext>(out_seal);
}

void ngraph::he::scalar_add(const HEPlaintext& arg0,
                            std::shared_ptr<HECiphertext>& arg1,
                            std::shared_ptr<HECiphertext>& out,
                            const element::Type& element_type,
                            const ngraph::he::HEBackend* he_backend) {
  ngraph::he::scalar_add(arg1, arg0, out, element_type, he_backend);
}
