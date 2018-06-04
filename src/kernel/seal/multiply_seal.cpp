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

#include "kernel/seal/multiply_seal.hpp"
#include "he_seal_backend.hpp"
#include "seal/seal.h"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::seal::scalar_multiply(
    const shared_ptr<runtime::he::SealCiphertextWrapper>& arg0,
    const shared_ptr<runtime::he::SealCiphertextWrapper>& arg1,
    shared_ptr<runtime::he::SealCiphertextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_seal::HESealBackend> he_seal_backend)
{
    he_seal_backend->get_evaluator()->multiply(
        arg0->m_ciphertext, arg1->m_ciphertext, out->m_ciphertext);
}

void runtime::he::kernel::seal::scalar_multiply(
    const shared_ptr<runtime::he::SealCiphertextWrapper>& arg0,
    const shared_ptr<runtime::he::SealPlaintextWrapper>& arg1,
    shared_ptr<runtime::he::SealCiphertextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_seal::HESealBackend> he_seal_backend)
{
    const string type_name = type.c_type_string();
    if (type_name == "float")
    {
        shared_ptr<runtime::he::HEPlaintext> p = he_seal_backend->create_empty_plaintext();

        if (arg1->m_plaintext == he_seal_backend->get_plaintext_num().fl_1)
        {
            out->m_ciphertext = arg0->m_ciphertext;
        }
        else if (arg1->m_plaintext == he_seal_backend->get_plaintext_num().fl_n1)
        {
            auto c = arg0->m_ciphertext;
            he_seal_backend->get_evaluator()->negate(c);
            out->m_ciphertext = c;
        }
        else if (arg1->m_plaintext == he_seal_backend->get_plaintext_num().fl_0)
        {
            float zero = 0;
            shared_ptr<runtime::he::HECiphertext> c = he_seal_backend->create_empty_ciphertext();
            shared_ptr<runtime::he::HEPlaintext> p = he_seal_backend->create_empty_plaintext();
            auto p_seal = dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(p);

            he_seal_backend->encode(p, &zero, type);
            he_seal_backend->encrypt(c, p);
            auto c_seal = dynamic_pointer_cast<runtime::he::SealCiphertextWrapper>(c);
            out = c_seal;
        }
        else
        {
            he_seal_backend->get_evaluator()->multiply_plain(
                arg0->m_ciphertext, arg1->m_plaintext, out->m_ciphertext);
        }
    }
    else if (type_name == "int64_t")
    {
        if (arg1->m_plaintext == he_seal_backend->get_plaintext_num().int64_1)
        {
            out->m_ciphertext = arg0->m_ciphertext;
        }
        else if (arg1->m_plaintext == he_seal_backend->get_plaintext_num().int64_n1)
        {
            auto c = arg0->m_ciphertext;
            he_seal_backend->get_evaluator()->negate(c);
            out->m_ciphertext = c;
        }
        else if (arg1->m_plaintext == he_seal_backend->get_plaintext_num().int64_0)
        {
            int zero = 0;
            shared_ptr<runtime::he::HECiphertext> c = he_seal_backend->create_empty_ciphertext();
            shared_ptr<runtime::he::HEPlaintext> p = he_seal_backend->create_empty_plaintext();
            auto p_seal = dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(p);

            he_seal_backend->encode(p, &zero, type);
            he_seal_backend->encrypt(c, p);
            auto c_seal = dynamic_pointer_cast<runtime::he::SealCiphertextWrapper>(c);
            out = c_seal;
        }
        else
        {
            he_seal_backend->get_evaluator()->multiply_plain(
                arg0->m_ciphertext, arg1->m_plaintext, out->m_ciphertext);
        }
    }
    else
    {
        throw ngraph_error("Multiply type not supported " + type_name);
    }
}

void runtime::he::kernel::seal::scalar_multiply(
    const shared_ptr<runtime::he::SealPlaintextWrapper>& arg0,
    const shared_ptr<runtime::he::SealCiphertextWrapper>& arg1,
    shared_ptr<runtime::he::SealCiphertextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_seal::HESealBackend> he_seal_backend)
{
    scalar_multiply(arg1, arg0, out, type, he_seal_backend);
}

void runtime::he::kernel::seal::scalar_multiply(
    const shared_ptr<runtime::he::SealPlaintextWrapper>& arg0,
    const shared_ptr<runtime::he::SealPlaintextWrapper>& arg1,
    shared_ptr<runtime::he::SealPlaintextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_seal::HESealBackend> he_seal_backend)
{
    const string type_name = type.c_type_string();
    if (type_name != "float")
    {
        throw ngraph_error("Type " + type_name + " not supported");
    }

    float x, y;
    he_seal_backend->decode(&x, arg0, type);
    he_seal_backend->decode(&y, arg1, type);
    float r = x * y;
    shared_ptr<runtime::he::HEPlaintext> out_he =
        dynamic_pointer_cast<runtime::he::HEPlaintext>(out);
    he_seal_backend->encode(out_he, &r, type);
    out = dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(out_he);
}
