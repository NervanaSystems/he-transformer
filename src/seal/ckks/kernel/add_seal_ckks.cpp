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

#include <iomanip>
#include <utility>

#include "seal/ckks/kernel/add_seal_ckks.hpp"
#include "seal/ckks/seal_ckks_util.hpp"

void ngraph::he::scalar_add_ckks(
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg0,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  NGRAPH_CHECK(arg0->complex_packing() == arg1->complex_packing());

  match_modulus_and_scale_inplace(arg0.get(), arg1.get(), he_seal_ckks_backend,
                                  pool);
  he_seal_ckks_backend->get_evaluator()->add(
      arg0->m_ciphertext, arg1->m_ciphertext, out->m_ciphertext);
  out->set_complex_packing(arg1->complex_packing());
}

void ngraph::he::scalar_add_ckks(
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg0,
    std::shared_ptr<ngraph::he::SealPlaintextWrapper>& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  if (arg1->is_single_value()) {
    float value = arg1->get_values()[0];
    double double_val = double(value);
    add_plain(arg0->m_ciphertext, double_val, out->m_ciphertext,
              he_seal_ckks_backend);
  } else {
    if (!arg1->is_encoded()) {
      // Just-in-time encoding at the right scale and modulus
      he_seal_ckks_backend->encode(arg1, arg0->m_ciphertext.parms_id(),
                                   arg0->m_ciphertext.scale(),
                                   arg0->complex_packing());
    } else {
      // Shouldn't be needed?
      // match_modulus_inplace(arg0.get(), arg1.get(), he_seal_ckks_backend,
      // pool);
      match_scale(arg0.get(), arg1.get(), he_seal_ckks_backend);
    }
    size_t chain_ind0 = get_chain_index(arg0.get(), he_seal_ckks_backend);
    size_t chain_ind1 = get_chain_index(arg1.get(), he_seal_ckks_backend);
    NGRAPH_CHECK(chain_ind0 == chain_ind1, "Chain inds ", chain_ind0, ",  ",
                 chain_ind1, " don't match");

    he_seal_ckks_backend->get_evaluator()->add_plain(
        arg0->m_ciphertext, arg1->get_plaintext(), out->m_ciphertext);
    out->set_complex_packing(arg0->complex_packing());
  }
}

void ngraph::he::scalar_add_ckks(
    std::shared_ptr<ngraph::he::SealPlaintextWrapper>& arg0,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  ngraph::he::scalar_add_ckks(arg1, arg0, out, element_type,
                              he_seal_ckks_backend, pool);
}
