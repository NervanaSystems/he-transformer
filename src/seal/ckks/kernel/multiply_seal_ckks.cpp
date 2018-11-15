//*****************************************************************************
// Copyright 2018 Intel Corporation
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

using namespace std;
using namespace ngraph::runtime::he;

void he_seal::ckks::kernel::scalar_multiply_ckks(
    const shared_ptr<const he_seal::SealCiphertextWrapper>& arg0,
    const shared_ptr<const he_seal::SealCiphertextWrapper>& arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const runtime::he::he_seal::HESealCKKSBackend* he_seal_ckks_backend) {
  // TODO: enable for different chain inds / scales.
  if ((arg0 == arg1) && (arg1 == out)) {
    he_seal_ckks_backend->get_evaluator()->square_inplace(out->m_ciphertext);
  } else if (arg1 == arg0) {
    he_seal_ckks_backend->get_evaluator()->square(arg1->m_ciphertext,
                                                  out->m_ciphertext);
  } else if (arg0 == out) {
    he_seal_ckks_backend->get_evaluator()->multiply_inplace(out->m_ciphertext,
                                                            arg1->m_ciphertext);
  } else if (arg1 == out) {
    he_seal_ckks_backend->get_evaluator()->multiply_inplace(out->m_ciphertext,
                                                            arg0->m_ciphertext);
  } else {
    he_seal_ckks_backend->get_evaluator()->multiply(
        arg0->m_ciphertext, arg1->m_ciphertext, out->m_ciphertext);
  }

  he_seal_ckks_backend->get_evaluator()->relinearize_inplace(
      out->m_ciphertext, *(he_seal_ckks_backend->get_relin_keys()));

  he_seal_ckks_backend->get_evaluator()->rescale_to_next_inplace(
      out->m_ciphertext);
}

void he_seal::ckks::kernel::scalar_multiply_ckks(
    const shared_ptr<const he_seal::SealCiphertextWrapper>& arg0,
    const shared_ptr<const he_seal::SealPlaintextWrapper>& arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const runtime::he::he_seal::HESealCKKSBackend* he_seal_ckks_backend) {
  auto arg0_scaled =
      make_shared<he_seal::SealCiphertextWrapper>(arg0->m_ciphertext);
  auto arg1_scaled =
      make_shared<he_seal::SealPlaintextWrapper>(arg1->m_plaintext);

  size_t chain_ind0 = he_seal_ckks_backend->get_context()
                          ->context_data(arg0->m_ciphertext.parms_id())
                          ->chain_index();

  size_t chain_ind1 = he_seal_ckks_backend->get_context()
                          ->context_data(arg1->m_plaintext.parms_id())
                          ->chain_index();

  if (chain_ind0 > chain_ind1) {
    he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
        arg0_scaled->m_ciphertext, arg1_scaled->m_plaintext.parms_id());
    chain_ind0 = he_seal_ckks_backend->get_context()
                     ->context_data(arg0_scaled->m_ciphertext.parms_id())
                     ->chain_index();
  } else if (chain_ind1 > chain_ind0) {
    he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
        arg1_scaled->m_plaintext, arg0_scaled->m_ciphertext.parms_id());
    chain_ind1 = he_seal_ckks_backend->get_context()
                     ->context_data(arg1_scaled->m_plaintext.parms_id())
                     ->chain_index();
  }
  NGRAPH_ASSERT(chain_ind0 == chain_ind1) << "Chain moduli do not match";

  if (arg0 == out) {
    he_seal_ckks_backend->get_evaluator()->multiply_plain_inplace(
        arg0_scaled->m_ciphertext, arg1_scaled->m_plaintext);
    out = arg0_scaled;
  } else {
    he_seal_ckks_backend->get_evaluator()->multiply_plain(
        arg0_scaled->m_ciphertext, arg1_scaled->m_plaintext, out->m_ciphertext);
  }

  he_seal_ckks_backend->get_evaluator()->relinearize_inplace(
      out->m_ciphertext, *(he_seal_ckks_backend->get_relin_keys()));

  // TODO: rescale only if needed? Check mod switching?
  // NGRAPH_DEBUG isn't thread-safe until ngraph commit #1977
  // https://github.com/NervanaSystems/ngraph/commit/ee6444ed39864776c8ce9a406eee9275382a88bb
  // so we comment it out.
  // TODO: uncomment at next ngraph version
  // NGRAPH_DEBUG << "Rescaling to next in place";
  he_seal_ckks_backend->get_evaluator()->rescale_to_next_inplace(
      out->m_ciphertext);
}

void he_seal::ckks::kernel::scalar_multiply_ckks(
    const shared_ptr<const he_seal::SealPlaintextWrapper>& arg0,
    const shared_ptr<const he_seal::SealCiphertextWrapper>& arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const runtime::he::he_seal::HESealCKKSBackend* he_seal_ckks_backend) {
  scalar_multiply_ckks(arg1, arg0, out, element_type, he_seal_ckks_backend);
}