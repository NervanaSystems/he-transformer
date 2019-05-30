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

#include "seal/kernel/add_seal.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/ckks/kernel/add_seal_ckks.hpp"

void ngraph::he::scalar_add(
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg0,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  if (arg0->is_zero() && arg1->is_zero()) {
    out->set_zero(true);
  }
  if (arg0->is_zero()) {
    out = std::make_shared<ngraph::he::SealCiphertextWrapper>(*arg1);
  } else if (arg1->is_zero()) {
    out = std::make_shared<ngraph::he::SealCiphertextWrapper>(*arg0);
  } else {
    auto const he_seal_ckks_backend =
        cast_to_seal_ckks_backend(he_seal_backend);
    ngraph::he::scalar_add_ckks(arg0, arg1, out, element_type,
                                he_seal_ckks_backend, pool);
  }
}

void ngraph::he::scalar_add(
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg0,
    std::shared_ptr<SealPlaintextWrapper>& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  NGRAPH_CHECK(element_type == element::f32);

  if (arg0->is_zero()) {
    auto arg1_hetext = std::dynamic_pointer_cast<ngraph::he::HEPlaintext>(arg1);
    auto out_hetext = std::dynamic_pointer_cast<ngraph::he::HECiphertext>(out);
    he_seal_backend->encrypt(out_hetext, arg1_hetext);
    out = ngraph::he::cast_to_seal_hetext(out_hetext);
    return;
  }

  // TODO: handle case where arg1 = {0, 0, 0, 0, ...}
  bool add_zero = arg1->is_single_value() && (arg1->get_values()[0] == 0.0f);

  if (add_zero) {
    out = std::make_shared<ngraph::he::SealCiphertextWrapper>(*arg0);
  } else {
    auto he_seal_ckks_backend = cast_to_seal_ckks_backend(he_seal_backend);
    ngraph::he::scalar_add_ckks(arg0, arg1, out, element_type,
                                he_seal_ckks_backend, pool);
  }
}

void ngraph::he::scalar_add(
    std::shared_ptr<SealPlaintextWrapper>& arg0,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  ngraph::he::scalar_add(arg1, arg0, out, element_type, he_seal_backend);
}

void ngraph::he::scalar_add(std::shared_ptr<SealPlaintextWrapper>& arg0,
                            std::shared_ptr<SealPlaintextWrapper>& arg1,
                            std::shared_ptr<SealPlaintextWrapper>& out,
                            const element::Type& element_type,
                            const HESealBackend* he_seal_backend,
                            const seal::MemoryPoolHandle& pool) {
  NGRAPH_CHECK(element_type == element::f32);

  const std::vector<float>& arg0_vals = arg0->get_values();
  const std::vector<float>& arg1_vals = arg1->get_values();
  std::vector<float> out_vals(arg0->num_values());

  std::transform(arg0_vals.begin(), arg0_vals.end(), arg1_vals.begin(),
                 out_vals.begin(), std::plus<float>());
  out->set_values(out_vals);
}
