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

void he_seal::ckks::kernel::scalar_add_ckks(
    he_seal::SealCiphertextWrapper* arg0, he_seal::SealCiphertextWrapper* arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  match_modulus_inplace(arg0, arg1, he_seal_ckks_backend, pool);
  match_scale(arg0, arg1, he_seal_ckks_backend);

  he_seal_ckks_backend->get_evaluator()->add(
      arg0->m_ciphertext, arg1->m_ciphertext, out->m_ciphertext);

  NGRAPH_ASSERT(arg0->complex_packing() == arg1->complex_packing());

  out->set_complex_packing(arg1->complex_packing());
  // NGRAPH_INFO << "Add output complex? " << arg1->complex_packing();
}

void he_seal::ckks::kernel::scalar_add_ckks(
    he_seal::SealCiphertextWrapper* arg0, he_seal::SealPlaintextWrapper* arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  /*if (arg0->complex_packing()) {
    NGRAPH_INFO << "Adding complex!";
  } else {
    NGRAPH_INFO << "Adding real";
  } */
  if (!arg1->is_encoded()) {
    // NGRAPH_INFO << "Encoding plaintext add";
    /*if (arg0->complex_packing()) {
      NGRAPH_INFO << "Encoding complex";
    } */
    he_seal_ckks_backend->encode(arg1, arg0->complex_packing());
  }

  match_modulus_inplace(arg0, arg1, he_seal_ckks_backend, pool);
  match_scale(arg0, arg1, he_seal_ckks_backend);

  he_seal_ckks_backend->get_evaluator()->add_plain(
      arg0->m_ciphertext, arg1->get_plaintext(), out->m_ciphertext);
  out->set_complex_packing(arg0->complex_packing());
  // NGRAPH_INFO << "Add output complex? " << arg0->complex_packing();
}

void he_seal::ckks::kernel::scalar_add_ckks(
    he_seal::SealPlaintextWrapper* arg0, he_seal::SealCiphertextWrapper* arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  he_seal::ckks::kernel::scalar_add_ckks(arg1, arg0, out, element_type,
                                         he_seal_ckks_backend, pool);
}
