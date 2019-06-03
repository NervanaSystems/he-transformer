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
#include "seal/he_seal_backend.hpp"
#include "seal/seal_util.hpp"

void ngraph::he::scalar_add_seal(
    ngraph::he::SealCiphertextWrapper& arg0,
    ngraph::he::SealCiphertextWrapper& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  if (arg0.is_zero() && arg0.is_zero()) {
    out->set_zero(true);
  } else if (arg0.is_zero()) {
    out = std::make_shared<ngraph::he::SealCiphertextWrapper>(arg1);
  } else if (arg1.is_zero()) {
    out = std::make_shared<ngraph::he::SealCiphertextWrapper>(arg0);
  } else {
    NGRAPH_CHECK(arg0.complex_packing() == arg1.complex_packing());

    match_modulus_and_scale_inplace(arg0, arg1, he_seal_backend, pool);
    he_seal_backend->get_evaluator()->add(arg0.ciphertext(), arg1.ciphertext(),
                                          out->ciphertext());
    out->set_complex_packing(arg1.complex_packing());
  }
}

void ngraph::he::scalar_add_seal(
    ngraph::he::SealCiphertextWrapper& arg0, const HEPlaintext& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  NGRAPH_CHECK(element_type == element::f32);

  if (arg0.is_zero()) {
    he_seal_backend->encrypt(out, arg1);
    return;
  }

  // TODO: handle case where arg1 = {0, 0, 0, 0, ...}
  bool add_zero = arg1.is_single_value() && (arg1.get_values()[0] == 0.0f);

  if (add_zero) {
    out = std::make_shared<ngraph::he::SealCiphertextWrapper>(arg0);
  } else {
    // NGRAPH_CHECK(arg1.complex_packing() == arg0.complex_packing(),
    //             "cipher/plain complex packing args differ");

    if (arg1.is_single_value()) {
      float value = arg1.get_values()[0];
      double double_val = double(value);
      add_plain(arg0.ciphertext(), double_val, out->ciphertext(),
                he_seal_backend);
    } else {
      auto p = SealPlaintextWrapper(arg1.complex_packing());
      he_seal_backend->encode(p, arg1, arg0.ciphertext().parms_id(),
                              arg0.ciphertext().scale());
      size_t chain_ind0 = get_chain_index(arg0, he_seal_backend);
      size_t chain_ind1 = get_chain_index(p.plaintext(), he_seal_backend);
      NGRAPH_CHECK(chain_ind0 == chain_ind1, "Chain inds ", chain_ind0, ",  ",
                   chain_ind1, " don't match");

      he_seal_backend->get_evaluator()->add_plain(
          arg0.ciphertext(), p.plaintext(), out->ciphertext());
      out->set_complex_packing(arg0.complex_packing());
    }
  }
}

void ngraph::he::scalar_add_seal(
    const HEPlaintext& arg0, ngraph::he::SealCiphertextWrapper& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  ngraph::he::scalar_add_seal(arg1, arg0, out, element_type, he_seal_backend);
}

void ngraph::he::scalar_add_seal(const HEPlaintext& arg0,
                                 const HEPlaintext& arg1, HEPlaintext& out,
                                 const element::Type& element_type,
                                 const HESealBackend* he_seal_backend,
                                 const seal::MemoryPoolHandle& pool) {
  NGRAPH_CHECK(element_type == element::f32);

  const std::vector<float>& arg0_vals = arg0.get_values();
  const std::vector<float>& arg1_vals = arg1.get_values();
  std::vector<float> out_vals(arg0.num_values());

  std::transform(arg0_vals.begin(), arg0_vals.end(), arg1_vals.begin(),
                 out_vals.begin(), std::plus<float>());
  out.set_values(out_vals);
}
