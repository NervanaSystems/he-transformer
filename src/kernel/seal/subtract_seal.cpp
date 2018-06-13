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

#include "kernel/seal/subtract_seal.hpp"
#include "he_seal_backend.hpp"
#include "kernel/seal/add_seal.hpp"
#include "kernel/seal/negate_seal.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::seal::scalar_subtract(
    const shared_ptr<runtime::he::SealCiphertextWrapper>& arg0,
    const shared_ptr<runtime::he::SealCiphertextWrapper>& arg1,
    shared_ptr<runtime::he::SealCiphertextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_seal::HESealBackend> he_seal_backend)
{
    he_seal_backend->get_evaluator()->sub(
        arg0->m_ciphertext, arg1->m_ciphertext, out->m_ciphertext);
}

void runtime::he::kernel::seal::scalar_subtract(
    const shared_ptr<runtime::he::SealPlaintextWrapper>& arg0,
    const shared_ptr<runtime::he::SealPlaintextWrapper>& arg1,
    shared_ptr<runtime::he::SealPlaintextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_seal::HESealBackend> he_seal_backend)
{
    shared_ptr<runtime::he::HEPlaintext> out_he =
        dynamic_pointer_cast<runtime::he::HEPlaintext>(out);
    const string type_name = type.c_type_string();
    if (type_name == "float")
    {
        float x, y;
        he_seal_backend->decode(&x, arg0, type);
        he_seal_backend->decode(&y, arg1, type);
        float r = x - y;
        he_seal_backend->encode(out_he, &r, type);
    }
    else if (type_name == "int64_t")
    {
        int64_t x, y;
        he_seal_backend->decode(&x, arg0, type);
        he_seal_backend->decode(&y, arg1, type);
        int64_t r = x - y;
        he_seal_backend->encode(out_he, &r, type);
    }
    else
    {
        throw ngraph_error("Unsupported type " + type_name + " in subtract");
    }
    out = dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(out_he);
}

void runtime::he::kernel::seal::scalar_subtract(
    const shared_ptr<runtime::he::SealCiphertextWrapper>& arg0,
    const shared_ptr<runtime::he::SealPlaintextWrapper>& arg1,
    shared_ptr<runtime::he::SealCiphertextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_seal::HESealBackend> he_seal_backend)
{
    const string type_name = type.c_type_string();
    if ((type_name == "float" && arg1->m_plaintext == he_seal_backend->get_plaintext_num().fl_0) ||
        (type_name == "int64_t" && arg1->m_plaintext == he_seal_backend->get_plaintext_num().int64_0))
    {
        out = arg0;
    }
    else
    {
        he_seal_backend->get_evaluator()->sub_plain(
                arg0->m_ciphertext, arg1->m_plaintext, out->m_ciphertext);
    }
}

void runtime::he::kernel::seal::scalar_subtract(
    const shared_ptr<runtime::he::SealPlaintextWrapper>& arg0,
    const shared_ptr<runtime::he::SealCiphertextWrapper>& arg1,
    shared_ptr<runtime::he::SealCiphertextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_seal::HESealBackend> he_seal_backend)
{
    he_seal_backend->get_evaluator()->negate(arg1->m_ciphertext, out->m_ciphertext);
    he_seal_backend->get_evaluator()->add_plain(
        out->m_ciphertext, arg0->m_plaintext, out->m_ciphertext);
}
