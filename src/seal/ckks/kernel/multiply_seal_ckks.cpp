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

#include "seal/ckks/kernel/multiply_seal_ckks.hpp"
#include "seal/ckks/seal_ckks_util.hpp"
#include "seal/seal.h"

void ngraph::he::scalar_multiply_ckks(
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg0,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const ngraph::he::HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  match_modulus_and_scale_inplace(arg0.get(), arg1.get(), he_seal_ckks_backend,
                                  pool);
  match_scale(arg0.get(), arg1.get(), he_seal_ckks_backend);
  size_t chain_ind0 = get_chain_index(arg0.get(), he_seal_ckks_backend);
  size_t chain_ind1 = get_chain_index(arg1.get(), he_seal_ckks_backend);

  if (chain_ind0 == 0 || chain_ind1 == 0) {
    NGRAPH_INFO << "Multiplicative depth limit reached";
    exit(1);
  }

  if (arg0 == arg1) {
    he_seal_ckks_backend->get_evaluator()->square(arg0->m_ciphertext,
                                                  out->m_ciphertext, pool);
  } else {
    he_seal_ckks_backend->get_evaluator()->multiply(
        arg0->m_ciphertext, arg1->m_ciphertext, out->m_ciphertext, pool);
  }

  he_seal_ckks_backend->get_evaluator()->relinearize_inplace(
      out->m_ciphertext, *(he_seal_ckks_backend->get_relin_keys()), pool);

  he_seal_ckks_backend->get_evaluator()->rescale_to_next_inplace(
      out->m_ciphertext, pool);
}

void ngraph::he::scalar_multiply_ckks(
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg0,
    std::shared_ptr<ngraph::he::HEPlaintext>& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const ngraph::he::HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  // TODO: activate!
  if (arg1->is_single_value()) {
    float value = arg1->get_values()[0];
    double double_val = double(value);
    multiply_plain(arg0->m_ciphertext, double_val, out->m_ciphertext,
                   he_seal_ckks_backend, pool);
  } else {
    if (!arg1->is_encoded()) {
      // Just-in-time encoding at the right scale and modulus
      he_seal_ckks_backend->encode(arg1, arg0->m_ciphertext.parms_id(),
                                   arg0->m_ciphertext.scale(), false);
    } else {
      // Shouldn't need to match modulus unless encoding went wrong
      // match_modulus_inplace(arg0.get(), arg1.get(), he_seal_ckks_backend,
      // pool);
      match_scale(arg0.get(), arg1.get(), he_seal_ckks_backend);
    }
    size_t chain_ind0 = get_chain_index(arg0.get(), he_seal_ckks_backend);
    size_t chain_ind1 = get_chain_index(arg1.get(), he_seal_ckks_backend);

    NGRAPH_CHECK(chain_ind0 == chain_ind1, "Chain_ind0 ", chain_ind0,
                 " != chain_ind1 ", chain_ind1);
    NGRAPH_CHECK(chain_ind0 > 0, "Multiplicative depth exceeded for arg0");
    NGRAPH_CHECK(chain_ind1 > 0, "Multiplicative depth exceeded for arg1");

    try {
      he_seal_ckks_backend->get_evaluator()->multiply_plain(
          arg0->m_ciphertext, arg1->get_plaintext(), out->m_ciphertext, pool);
    } catch (const std::exception& e) {
      NGRAPH_INFO << "Error multiplying plain " << e.what();
      NGRAPH_INFO << "arg1->get_values().size() " << arg1->get_values().size();
      auto& values = arg1->get_values();
      for (const auto& elem : values) {
        NGRAPH_INFO << elem;
      }
    }
  }
  out->set_complex_packing(arg0->complex_packing());
  // NGRAPH_INFO << "Skipping relin and rescale!";

  // Don't relinearize after plain multiply!
  // he_seal_ckks_backend->get_evaluator()->relinearize_inplace(
  //    out->m_ciphertext, *(he_seal_ckks_backend->get_relin_keys()), pool);

  // Don't rescale after every mult! Only after dot / conv
  // he_seal_ckks_backend->get_evaluator()->rescale_to_next_inplace(
  //    out->m_ciphertext, pool);
}

void ngraph::he::scalar_multiply_ckks(
    std::shared_ptr<ngraph::he::HEPlaintext>& arg0,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const ngraph::he::HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  ngraph::he::scalar_multiply_ckks(arg1, arg0, out, element_type,
                                   he_seal_ckks_backend, pool);
}
