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

#include <iomanip>
#include <utility>

#include "seal/ckks/kernel/add_seal_ckks.hpp"
#include "seal_ckks_util.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void he_seal::ckks::kernel::scalar_add_ckks(
    const shared_ptr<const he_seal::SealCiphertextWrapper>& arg0,
    const shared_ptr<const he_seal::SealCiphertextWrapper>& arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealCKKSBackend* he_seal_ckks_backend) {
  auto argument_matching_pair =
      match_arguments(arg0, arg1, he_seal_ckks_backend);
  auto arg0_scaled = get<0>(argument_matching_pair);
  auto arg1_scaled = get<1>(argument_matching_pair);

  he_seal_ckks_backend->get_evaluator()->add(
      arg0_scaled->m_ciphertext, arg1_scaled->m_ciphertext, out->m_ciphertext);
}

void he_seal::ckks::kernel::scalar_add_ckks(
    const shared_ptr<const he_seal::SealCiphertextWrapper>& arg0,
    const shared_ptr<const he_seal::SealPlaintextWrapper>& arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealCKKSBackend* he_seal_ckks_backend) {
  auto argument_matching_pair =
      match_arguments(arg0, arg1, he_seal_ckks_backend);
  auto arg0_scaled = get<0>(argument_matching_pair);
  auto arg1_scaled = get<1>(argument_matching_pair);

  he_seal_ckks_backend->get_evaluator()->add_plain(
      arg0_scaled->m_ciphertext, arg1_scaled->m_plaintext, out->m_ciphertext);
}

void he_seal::ckks::kernel::scalar_add_ckks(
    const shared_ptr<const he_seal::SealPlaintextWrapper>& arg0,
    const shared_ptr<const he_seal::SealCiphertextWrapper>& arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealCKKSBackend* he_seal_ckks_backend) {
  he_seal::ckks::kernel::scalar_add_ckks(arg1, arg0, out, element_type,
                                         he_seal_ckks_backend);
}
