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

#include "seal/kernel/negate_seal.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void he_seal::kernel::scalar_negate(
    shared_ptr<he_seal::SealCiphertextWrapper>& arg,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend) {
  NGRAPH_ASSERT(element_type == element::f32);

  if (arg->is_zero()) {
    NGRAPH_INFO << "Arg is 0 in negate(C)";
    out->set_zero(true);
    return;
  }

  he_seal_backend->get_evaluator()->negate(arg->m_ciphertext,
                                           out->m_ciphertext);
}

void he_seal::kernel::scalar_negate(
    shared_ptr<he_seal::SealPlaintextWrapper>& arg,
    shared_ptr<he_seal::SealPlaintextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend) {
  NGRAPH_ASSERT(element_type == element::f32);

  const std::vector<float>& arg_vals = arg->get_values();
  std::vector<float> out_vals(arg->num_values());

  std::transform(arg_vals.begin(), arg_vals.end(), out_vals.begin(),
                 std::negate<float>());
  out->set_values(out_vals);
}