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

#include "seal/bfv/kernel/add_seal_bfv.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void he_seal::bfv::kernel::scalar_add_bfv(
    shared_ptr<he_seal::SealCiphertextWrapper>& arg0,
    shared_ptr<he_seal::SealCiphertextWrapper>& arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBFVBackend* he_seal_bfv_backend) {
  if (arg0.get() == out.get()) {
    he_seal_bfv_backend->get_evaluator()->add_inplace(out->m_ciphertext,
                                                      arg1->m_ciphertext);
  } else if (arg1.get() == out.get()) {
    he_seal_bfv_backend->get_evaluator()->add_inplace(out->m_ciphertext,
                                                      arg0->m_ciphertext);
  } else {
    he_seal_bfv_backend->get_evaluator()->add(
        arg0->m_ciphertext, arg1->m_ciphertext, out->m_ciphertext);
  }
}

void he_seal::bfv::kernel::scalar_add_bfv(
    shared_ptr<he_seal::SealCiphertextWrapper>& arg0,
    shared_ptr<he_seal::SealPlaintextWrapper>& arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBFVBackend* he_seal_bfv_backend) {
  if (!arg1->is_encoded()) {
    he_seal_bfv_backend->encode(arg1, false);
  }
  if (arg0.get() == out.get()) {
    he_seal_bfv_backend->get_evaluator()->add_plain_inplace(
        out->m_ciphertext, arg1->get_plaintext());
  } else {
    he_seal_bfv_backend->get_evaluator()->add_plain(
        arg0->m_ciphertext, arg1->get_plaintext(), out->m_ciphertext);
  }
}

void he_seal::bfv::kernel::scalar_add_bfv(
    shared_ptr<he_seal::SealPlaintextWrapper>& arg0,
    shared_ptr<he_seal::SealCiphertextWrapper>& arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBFVBackend* he_seal_bfv_backend) {
  he_seal::bfv::kernel::scalar_add_bfv(arg1, arg0, out, element_type,
                                       he_seal_bfv_backend);
}
