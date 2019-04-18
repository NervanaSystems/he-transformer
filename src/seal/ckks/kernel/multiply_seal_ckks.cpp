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

using namespace std;
using namespace ngraph::runtime::he;

void he_seal::ckks::kernel::scalar_multiply_ckks(
    he_seal::SealCiphertextWrapper* arg0, he_seal::SealCiphertextWrapper* arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const runtime::he::he_seal::HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {

  match_modulus_inplace(arg0, arg1, he_seal_ckks_backend, pool);
  match_scale(arg0, arg1, he_seal_ckks_backend);

  size_t chain_ind0 = he_seal_ckks_backend->get_context()
                          ->context_data(arg0->get_hetext().parms_id())
                          ->chain_index();
  size_t chain_ind1 = he_seal_ckks_backend->get_context()
                          ->context_data(arg1->get_hetext().parms_id())
                          ->chain_index();

  if (chain_ind0 != 2 || chain_ind1 != 2) {
    NGRAPH_INFO << "Chaind inds " << chain_ind0 << ", " << chain_ind1;
    exit(1);
  }

  match_modulus_inplace(arg0, arg1, he_seal_ckks_backend);
  chain_ind0 = he_seal_ckks_backend->get_context()
                   ->context_data(arg0->get_hetext().parms_id())
                   ->chain_index();
  chain_ind1 = he_seal_ckks_backend->get_context()
                   ->context_data(arg1->get_hetext().parms_id())
                   ->chain_index();

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

void he_seal::ckks::kernel::scalar_multiply_ckks(
    he_seal::SealCiphertextWrapper* arg0, he_seal::SealPlaintextWrapper* arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const runtime::he::he_seal::HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  match_modulus_inplace(arg0, arg1, he_seal_ckks_backend, pool);
  match_scale(arg0, arg1, he_seal_ckks_backend);

  size_t chain_ind0 = he_seal_ckks_backend->get_context()
                          ->context_data(arg0->get_hetext().parms_id())
                          ->chain_index();
  size_t chain_ind1 = he_seal_ckks_backend->get_context()
                          ->context_data(arg1->get_hetext().parms_id())
                          ->chain_index();

  if (chain_ind0 == 0 || chain_ind1 == 0) {
    NGRAPH_INFO << "Multiplicative depth limit reached";
    exit(1);
  }

  // NGRAPH_INFO << "arg1->get_plaintext() " << arg1->get_value();

  he_seal_ckks_backend->get_evaluator()->multiply_plain(
      arg0->m_ciphertext, arg1->get_plaintext(), out->m_ciphertext, pool);

  he_seal_ckks_backend->get_evaluator()->relinearize_inplace(
      out->m_ciphertext, *(he_seal_ckks_backend->get_relin_keys()), pool);

  he_seal_ckks_backend->get_evaluator()->rescale_to_next_inplace(
      out->m_ciphertext, pool);

  size_t chain_ind_out = he_seal_ckks_backend->get_context()
                             ->context_data(out->get_hetext().parms_id())
                             ->chain_index();

  if (chain_ind_out != 1) {
    NGRAPH_INFO << "Chain ind after mult: " << chain_ind_out;
    exit(1);
  }
}

void he_seal::ckks::kernel::scalar_multiply_ckks(
    he_seal::SealPlaintextWrapper* arg0, he_seal::SealCiphertextWrapper* arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const runtime::he::he_seal::HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  scalar_multiply_ckks(arg1, arg0, out, element_type, he_seal_ckks_backend,
                       pool);
}