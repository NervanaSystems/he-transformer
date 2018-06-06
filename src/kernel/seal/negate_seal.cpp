/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "kernel/seal/negate_seal.hpp"
#include "he_seal_backend.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::seal::scalar_negate(
    const shared_ptr<runtime::he::SealCiphertextWrapper>& arg0,
    shared_ptr<runtime::he::SealCiphertextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_seal::HESealBackend> he_seal_backend)
{
    he_seal_backend->get_evaluator()->negate(arg0->m_ciphertext, out->m_ciphertext);
}

void runtime::he::kernel::seal::scalar_negate(
    const shared_ptr<runtime::he::SealPlaintextWrapper>& arg0,
    shared_ptr<runtime::he::SealPlaintextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_seal::HESealBackend> he_seal_backend)
{
    const string type_name = type.c_type_string();
    if (type_name != "float")
    {
        throw ngraph_error("Unsupported type " + type_name + " in negate");
    }

    float x;
    he_seal_backend->decode(&x, arg0, type);
    float r = -x;
    shared_ptr<runtime::he::HEPlaintext> out_he =
        dynamic_pointer_cast<runtime::he::HEPlaintext>(out);
    he_seal_backend->encode(out_he, &r, type);
    out = dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(out_he);
}
