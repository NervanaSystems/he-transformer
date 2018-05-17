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
        seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::New(false);
        scalar_multiply(arg0[i], arg1[i], out[i], type, he_backend, pool);
    }
}

void runtime::he::kernel::scalar_multiply(const shared_ptr<he::HECiphertext>& arg0,
                                          const shared_ptr<he::HECiphertext>& arg1,
                                          shared_ptr<he::HECiphertext>& out,
                                          const element::Type& type,
                                          shared_ptr<HEBackend> he_backend,
                                          const seal::MemoryPoolHandle& pool)
{
    auto he_seal_backend = dynamic_pointer_cast<HESealBackend>(he_backend);
    if (!he_seal_backend)
    {
        throw ngraph_error("HE backend not seal type");
    }
    he_seal_backend->get_evaluator()->multiply(*arg0, *arg1, *out, pool);
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
        seal::MemoryPoolHandle pool = he::HEMemoryPoolHandle::New(false);
        scalar_multiply(arg0[i], arg1[i], out[i], type, he_backend, pool);
    }
}

void runtime::he::kernel::scalar_multiply(const shared_ptr<he::HECiphertext>& arg0,
                                          const shared_ptr<he::HEPlaintext>& arg1,
                                          shared_ptr<he::HECiphertext>& out,
                                          const element::Type& type,
                                          shared_ptr<HEBackend> he_backend,
                                          const he::HEMemoryPoolHandle& pool)
{
    const string type_name = type.c_type_string();
    if (type_name == "float")
    {
        if (*arg1 == he_backend->get_plaintext_num().fl_1)
        {
            *out = *arg0;
        }
        else if (*arg1 == he_backend->get_plaintext_num().fl_n1)
        {
            he::HECiphertext c = *arg0;
            he_backend->get_evaluator()->negate(c);
            *out = c;
        }
        else
        {
            he_backend->get_evaluator()->multiply_plain(*arg0, *arg1, *out, pool);
        }
    }
    else if (type_name == "int64_t")
    {
        if (*arg1 == he_backend->get_plaintext_num().fl_1)
        {
            *out = *arg0;
        }
        else if (*arg1 == he_backend->get_plaintext_num().fl_n1)
        {
            he::HECiphertext c = *arg0;
            he_backend->get_evaluator()->negate(c);
            *out = c;
        }
        else
        {
            he_backend->get_evaluator()->multiply_plain(*arg0, *arg1, *out, pool);
        }
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
                                          shared_ptr<HEBackend> he_backend,
                                          const seal::MemoryPoolHandle& pool)
{
    scalar_multiply(arg1, arg0, out, type, he_backend, pool);
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

#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        auto evaluator = he_backend->get_evaluator();
        float x, y;
        he_backend->decode(&x, *arg0[i], type);
        he_backend->decode(&y, *arg1[i], type);
        float r = x * y;
        he_backend->encode(*out[i], &r, type);
    }
}

void runtime::he::kernel::scalar_multiply(const shared_ptr<he::HEPlaintext>& arg0,
                                          const shared_ptr<he::HEPlaintext>& arg1,
                                          shared_ptr<he::HEPlaintext>& out,
                                          const element::Type& type,
                                          shared_ptr<HEBackend> he_backend,
                                          const seal::MemoryPoolHandle& pool)
{
    auto evaluator = he_backend->get_evaluator();
    float x, y;
    he_backend->decode(&x, *arg0, type);
    he_backend->decode(&y, *arg1, type);
    float r = x * y;
    he_backend->encode(*out, &r, type);
}
