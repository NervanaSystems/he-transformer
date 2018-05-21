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

#include <vector>

#include "he_backend.hpp"
#include "kernel/multiply.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/seal.h"
#include "he_seal_backend.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::multiply(const vector<shared_ptr<he::HECiphertext>>& arg0,
                                   const vector<shared_ptr<he::HECiphertext>>& arg1,
                                   vector<shared_ptr<he::HECiphertext>>& out,
                                   const element::Type& type,
                                   shared_ptr<HEBackend> he_backend,
                                   size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_multiply(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::scalar_multiply(const shared_ptr<he::HECiphertext>& arg0,
                                          const shared_ptr<he::HECiphertext>& arg1,
                                          shared_ptr<he::HECiphertext>& out,
                                          const element::Type& type,
                                          shared_ptr<HEBackend> he_backend)
{
    auto he_seal_backend = dynamic_pointer_cast<HESealBackend>(he_backend);
    if (!he_seal_backend)
    {
        throw ngraph_error("HE backend not seal type");
    }
    auto arg0_seal = dynamic_pointer_cast<SealCiphertextWrapper>(arg0);
    auto arg1_seal = dynamic_pointer_cast<SealCiphertextWrapper>(arg1);
    auto out_seal = dynamic_pointer_cast<SealCiphertextWrapper>(out);
    he_seal_backend->get_evaluator()->multiply(arg0_seal->m_ciphertext, arg1_seal->m_ciphertext, out_seal->m_ciphertext);
}

void runtime::he::kernel::multiply(const vector<shared_ptr<he::HECiphertext>>& arg0,
                                   const vector<shared_ptr<he::HEPlaintext>>& arg1,
                                   vector<shared_ptr<he::HECiphertext>>& out,
                                   const element::Type& type,
                                   shared_ptr<HEBackend> he_backend,
                                   size_t count)
{
     #pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_multiply(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::scalar_multiply(const shared_ptr<he::HECiphertext>& arg0,
                                          const shared_ptr<he::HEPlaintext>& arg1,
                                          shared_ptr<he::HECiphertext>& out,
                                          const element::Type& type,
                                          shared_ptr<HEBackend> he_backend)
{
    auto he_seal_backend = dynamic_pointer_cast<HESealBackend>(he_backend);
    if (!he_seal_backend)
    {
        throw ngraph_error("HE backend not seal type");
    }
    auto arg0_seal = dynamic_pointer_cast<SealCiphertextWrapper>(arg0);
    auto arg1_seal = dynamic_pointer_cast<SealPlaintextWrapper>(arg1);
    auto out_seal = dynamic_pointer_cast<SealCiphertextWrapper>(out);

    if (arg0_seal == nullptr || arg1_seal == nullptr || out_seal == nullptr)
    {
        throw ngraph_error("Non-seal arguments in scalar_multiply");
    }

    const string type_name = type.c_type_string();
    if (type_name == "float")
    {
        he_seal_backend->get_evaluator()->multiply_plain(arg0_seal->m_ciphertext, arg1_seal->m_plaintext, out_seal->m_ciphertext);
    }
    else if (type_name == "int64_t")
    {
        he_seal_backend->get_evaluator()->multiply_plain(arg0_seal->m_ciphertext, arg1_seal->m_plaintext, out_seal->m_ciphertext);
    }
    else
    {
        throw ngraph_error("Multiply type not supported " + type_name);
    }
}

void runtime::he::kernel::multiply(const vector<shared_ptr<he::HEPlaintext>>& arg0,
                                   const vector<shared_ptr<he::HECiphertext>>& arg1,
                                   vector<shared_ptr<he::HECiphertext>>& out,
                                   const element::Type& type,
                                   shared_ptr<HEBackend> he_backend,
                                   size_t count)
{
    multiply(arg1, arg0, out, type, he_backend, count);
}

void runtime::he::kernel::scalar_multiply(const shared_ptr<he::HEPlaintext>& arg0,
                                          const shared_ptr<he::HECiphertext>& arg1,
                                          shared_ptr<he::HECiphertext>& out,
                                          const element::Type& type,
                                          shared_ptr<HEBackend> he_backend)
{
    scalar_multiply(arg1, arg0, out, type, he_backend);
}

void runtime::he::kernel::multiply(const vector<shared_ptr<he::HEPlaintext>>& arg0,
                                   const vector<shared_ptr<he::HEPlaintext>>& arg1,
                                   vector<shared_ptr<he::HEPlaintext>>& out,
                                   const element::Type& type,
                                   shared_ptr<HEBackend> he_backend,
                                   size_t count)
{
    const string type_name = type.c_type_string();
    if (type_name != "float")
    {
        throw ngraph_error("Type " + type_name + " not supported");
    }

    auto he_seal_backend = dynamic_pointer_cast<HESealBackend>(he_backend);
    if (!he_seal_backend)
    {
        throw ngraph_error("HE backend not seal type");
    }

#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        float x, y;
        he_seal_backend->decode(&x, arg0[i], type);
        he_seal_backend->decode(&y, arg1[i], type);
        float r = x * y;
        he_seal_backend->encode(out[i], &r, type);
    }
}

void runtime::he::kernel::scalar_multiply(const shared_ptr<he::HEPlaintext>& arg0,
                                          const shared_ptr<he::HEPlaintext>& arg1,
                                          shared_ptr<he::HEPlaintext>& out,
                                          const element::Type& type,
                                          shared_ptr<HEBackend> he_backend)
{
    const string type_name = type.c_type_string();
    if (type_name != "float")
    {
        throw ngraph_error("Type " + type_name + " not supported");
    }

    auto he_seal_backend = dynamic_pointer_cast<HESealBackend>(he_backend);
    if (!he_seal_backend)
    {
        throw ngraph_error("HE backend not seal type");
    }
    float x, y;
    he_seal_backend->decode(&x, arg0, type);
    he_seal_backend->decode(&y, arg1, type);
    float r = x * y;
    he_seal_backend->encode(out, &r, type);
}
