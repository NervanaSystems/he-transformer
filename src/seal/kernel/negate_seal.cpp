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

#include "seal/kernel/negate_seal.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void he_seal::kernel::scalar_negate(
    const shared_ptr<const he_seal::SealCiphertextWrapper>& arg,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend) {
  if (arg == out) {
    he_seal_backend->get_evaluator()->negate_inplace(out->m_ciphertext);
  } else {
    he_seal_backend->get_evaluator()->negate(arg->m_ciphertext,
                                             out->m_ciphertext);
  }
}

void he_seal::kernel::scalar_negate(
    const shared_ptr<he_seal::SealPlaintextWrapper>& arg,
    shared_ptr<he_seal::SealPlaintextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend) {
  shared_ptr<HEPlaintext> out_he = dynamic_pointer_cast<HEPlaintext>(out);
  const string type_name = element_type.c_type_string();
  if (type_name == "float") {
    float x;
    he_seal_backend->decode(&x, arg.get(), element_type);
    float r = -x;
    he_seal_backend->encode(out_he, &r, element_type);
  } else {
    throw ngraph_error("Unsupported element type " + type_name + " in negate");
  }
  out = dynamic_pointer_cast<he_seal::SealPlaintextWrapper>(out_he);
}