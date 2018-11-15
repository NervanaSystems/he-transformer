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

#include "seal_ckks_util.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"

using namespace std;
using namespace ngraph::runtime::he;
/*
pair<shared_ptr<he_seal::SealCiphertextWrapper>,
     shared_ptr<he_seal::SealCiphertextWrapper>>
he_seal::ckks::kernel::match_arguments(
    const shared_ptr<const he_seal::SealCiphertextWrapper>& arg0,
    const shared_ptr<const he_seal::SealCiphertextWrapper>& arg1,
    const HESealCKKSBackend* he_seal_ckks_backend) {
  auto arg0_scaled =
      make_shared<he_seal::SealCiphertextWrapper>(arg0->m_ciphertext);
  auto arg1_scaled =
      make_shared<he_seal::SealCiphertextWrapper>(arg1->m_ciphertext);

  size_t chain_ind0 = he_seal_ckks_backend->get_context()
                          ->context_data(arg0->m_ciphertext.parms_id())
                          ->chain_index();

  size_t chain_ind1 = he_seal_ckks_backend->get_context()
                          ->context_data(arg1->m_ciphertext.parms_id())
                          ->chain_index();

  NGRAPH_INFO << "Chain inds " << chain_ind0 << ", " << chain_ind1;

  if (chain_ind0 > chain_ind1) {
    he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
        arg0_scaled->m_ciphertext, arg1_scaled->m_ciphertext.parms_id());
    chain_ind0 = he_seal_ckks_backend->get_context()
                     ->context_data(arg0_scaled->m_ciphertext.parms_id())
                     ->chain_index();
  } else if (chain_ind1 > chain_ind0) {
    he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
        arg1_scaled->m_ciphertext, arg0_scaled->m_ciphertext.parms_id());
    chain_ind1 = he_seal_ckks_backend->get_context()
                     ->context_data(arg1_scaled->m_ciphertext.parms_id())
                     ->chain_index();
    assert(chain_ind0 == chain_ind1);
  }

  return make_pair(arg0_scaled, arg1_scaled);
}
*/