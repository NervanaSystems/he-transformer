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

using namespace std;
using namespace ngraph::runtime::he;
using namespace ngraph::runtime::he::he_seal::ckks;

void he_seal::ckks::kernel::scalar_add_ckks(
    shared_ptr<he_seal::SealCiphertextWrapper>& arg0,
    shared_ptr<he_seal::SealCiphertextWrapper>& arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  NGRAPH_INFO << "Adding regular C+C=>C";
  NGRAPH_ASSERT(arg0->complex_packing() == arg1->complex_packing());
  NGRAPH_INFO << "Add chain inds before add matching";
  NGRAPH_INFO << "arg0: (" << get_chain_index(arg0.get(), he_seal_ckks_backend)
              << ", " << arg0->get_hetext().scale() << "), "
              << "arg1: (" << get_chain_index(arg1.get(), he_seal_ckks_backend)
              << ", " << arg1->get_hetext().scale() << ")";

  match_modulus_and_scale_inplace(arg0.get(), arg1.get(), he_seal_ckks_backend,
                                  pool);

  NGRAPH_INFO << "Add chain inds after add matching";
  NGRAPH_INFO << "arg0: (" << get_chain_index(arg0.get(), he_seal_ckks_backend)
              << ", " << arg0->get_hetext().scale() << "), "
              << "arg1: (" << get_chain_index(arg1.get(), he_seal_ckks_backend)
              << ", " << arg1->get_hetext().scale() << ")";

  he_seal_ckks_backend->get_evaluator()->add(
      arg0->m_ciphertext, arg1->m_ciphertext, out->m_ciphertext);
  out->set_complex_packing(arg1->complex_packing());

  NGRAPH_INFO << "Add chain inds after add";
  NGRAPH_INFO << "arg0: (" << get_chain_index(arg0.get(), he_seal_ckks_backend)
              << ", " << arg0->get_hetext().scale() << "), "
              << "arg1: (" << get_chain_index(arg1.get(), he_seal_ckks_backend)
              << ", " << arg1->get_hetext().scale() << ")";
}

void he_seal::ckks::kernel::scalar_add_ckks(
    shared_ptr<he_seal::SealCiphertextWrapper>& arg0,
    shared_ptr<he_seal::SealPlaintextWrapper>& arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
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

  NGRAPH_ASSERT(arg0->get_hetext().scale() == arg1->get_hetext().scale())
      << "arg0_scale " << arg0->get_hetext().scale() << " != arg1_scale "
      << arg1->get_hetext().scale();

  size_t chain_ind0 = get_chain_index(arg0.get(), he_seal_ckks_backend);
  size_t chain_ind1 = get_chain_index(arg1.get(), he_seal_ckks_backend);

  NGRAPH_ASSERT(chain_ind0 == chain_ind1)
      << "Chain_ind0 " << chain_ind0 << " != chain_ind1 " << chain_ind1;

  he_seal_ckks_backend->get_evaluator()->add_plain(
      arg0->m_ciphertext, arg1->get_plaintext(), out->m_ciphertext);
  out->set_complex_packing(arg0->complex_packing());
}

void he_seal::ckks::kernel::scalar_add_ckks(
    shared_ptr<he_seal::SealPlaintextWrapper>& arg0,
    shared_ptr<he_seal::SealCiphertextWrapper>& arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  he_seal::ckks::kernel::scalar_add_ckks(arg1, arg0, out, element_type,
                                         he_seal_ckks_backend, pool);
}
