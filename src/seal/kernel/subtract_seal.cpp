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

#include "seal/kernel/subtract_seal.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::he_seal::kernel::scalar_subtract(
    he_seal::SealCiphertextWrapper* arg0, he_seal::SealCiphertextWrapper* arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend) {
  if (arg0 == out.get()) {
    he_seal_backend->get_evaluator()->sub_inplace(out->m_ciphertext,
                                                  arg1->m_ciphertext);
  } else if (arg1 == out.get()) {
    he_seal_backend->get_evaluator()->sub_inplace(out->m_ciphertext,
                                                  arg0->m_ciphertext);
  } else {
    he_seal_backend->get_evaluator()->sub(
        arg0->m_ciphertext, arg1->m_ciphertext, out->m_ciphertext);
  }
}

void runtime::he::he_seal::kernel::scalar_subtract(
    he_seal::SealCiphertextWrapper* arg0, he_seal::SealPlaintextWrapper* arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend) {
  he_seal_backend->encode(arg1, arg0->complex_packing());
  if (arg0 == out.get()) {
    he_seal_backend->get_evaluator()->sub_plain_inplace(out->m_ciphertext,
                                                        arg1->get_plaintext());
  } else {
    he_seal_backend->get_evaluator()->sub_plain(
        arg0->m_ciphertext, arg1->get_plaintext(), out->m_ciphertext);
  }
}

void runtime::he::he_seal::kernel::scalar_subtract(
    he_seal::SealPlaintextWrapper* arg0, he_seal::SealPlaintextWrapper* arg1,
    shared_ptr<he_seal::SealPlaintextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend) {
  NGRAPH_ASSERT(element_type == element::f32);

  const std::vector<float>& arg0_vals = arg0->get_values();
  const std::vector<float>& arg1_vals = arg1->get_values();
  std::vector<float> out_vals(arg0->num_values());

  std::transform(arg0_vals.begin(), arg0_vals.end(), arg1_vals.begin(),
                 out_vals.begin(), std::minus<float>());
  out->set_values(out_vals);
}