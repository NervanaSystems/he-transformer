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
#include "kernel/add.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/seal.h"
#include "he_seal_backend.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::add(const vector<shared_ptr<he::HECiphertext>>& arg0,
                              const vector<shared_ptr<he::HECiphertext>>& arg1,
                              vector<shared_ptr<he::HECiphertext>>& out,
                              shared_ptr<HEBackend> he_backend,
                              size_t count)
{
    auto he_seal_backend = dynamic_pointer_cast<HESealBackend>(he_backend);
    if (!he_seal_backend)
    {
        throw ngraph_error("HE backend not seal type");
    }
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        auto arg0i = dynamic_pointer_cast<SealCiphertextWrapper>(arg0[i]);
        auto arg1i = dynamic_pointer_cast<SealCiphertextWrapper>(arg1[i]);
        auto outi = dynamic_pointer_cast<SealCiphertextWrapper>(out[i]);
        if (arg0i != nullptr && arg1i != nullptr && outi != nullptr)
        {
            he_seal_backend->get_evaluator()->add(arg0i->m_ciphertext, arg1i->m_ciphertext, outi->m_ciphertext);
        }
        else
        {
            throw ngraph_error("Add backend is seal, but arguments or outputs are not seal::Ciphertext");
        }
    }
}

void runtime::he::kernel::add(const vector<shared_ptr<he::HECiphertext>>& arg0,
                              const vector<shared_ptr<he::HEPlaintext>>& arg1,
                              vector<shared_ptr<he::HECiphertext>>& out,
                              shared_ptr<HEBackend> he_backend,
                              size_t count)
{
	auto he_seal_backend = dynamic_pointer_cast<HESealBackend>(he_backend);
	if (!he_seal_backend)
	{
		throw ngraph_error("HE backend not seal type");
	}
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        auto arg0i = dynamic_pointer_cast<SealCiphertextWrapper>(arg0[i]);
        auto arg1i = dynamic_pointer_cast<SealPlaintextWrapper>(arg1[i]);
        auto outi = dynamic_pointer_cast<SealCiphertextWrapper>(out[i]);
        if (arg0i != nullptr && arg1i != nullptr && outi != nullptr)
        {
            he_seal_backend->get_evaluator()->add_plain(arg0i->m_ciphertext, arg1i->m_plaintext, outi->m_ciphertext);
        }
        else
        {
            throw ngraph_error("Add backend is seal, but arguments or outputs are not seal::Ciphertext");
        }
    }
}

void runtime::he::kernel::add(const vector<shared_ptr<he::HEPlaintext>>& arg0,
                              const vector<shared_ptr<he::HECiphertext>>& arg1,
                              vector<shared_ptr<he::HECiphertext>>& out,
                              shared_ptr<HEBackend> he_backend,
                              size_t count)
{
    add(arg1, arg0, out, he_backend, count);
}

void runtime::he::kernel::add(const vector<shared_ptr<he::HEPlaintext>>& arg0,
                              const vector<shared_ptr<he::HEPlaintext>>& arg1,
                              vector<shared_ptr<he::HEPlaintext>>& out,
                              const element::Type& type,
                              shared_ptr<HEBackend> he_backend,
                              size_t count)
{
	auto he_seal_backend = dynamic_pointer_cast<HESealBackend>(he_backend);
	if (!he_seal_backend)
	{
		throw ngraph_error("HE backend not seal type");
	}
    const string type_name = type.c_type_string();
    if (type_name != "float")
    {
        throw ngraph_error("Unsupported type " + type_name + " in add");
    }
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        float x, y;
        he_seal_backend->decode(&x, arg0[i], type);
        he_seal_backend->decode(&y, arg1[i], type);
        float r = x + y;
        he_seal_backend->encode(out[i], &r, type);
    }
}

void runtime::he::kernel::scalar_add(const shared_ptr<he::HECiphertext>& arg0,
                                     const shared_ptr<he::HECiphertext>& arg1,
                                     shared_ptr<he::HECiphertext>& out,
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

    if (arg0_seal == nullptr || arg1_seal == nullptr || out_seal == nullptr)
    {
        throw ngraph_error("scalar_add receieved seal backend, but non-seal tensors");
    }
    he_seal_backend->get_evaluator()->add(arg0_seal->m_ciphertext, arg1_seal->m_ciphertext, out_seal->m_ciphertext);
}

void runtime::he::kernel::scalar_add(const shared_ptr<he::HEPlaintext>& arg0,
                                     const shared_ptr<he::HEPlaintext>& arg1,
                                     shared_ptr<he::HEPlaintext>& out,
                                     const element::Type& type,
                                     shared_ptr<HEBackend> he_backend)
{
    auto he_seal_backend = dynamic_pointer_cast<HESealBackend>(he_backend);
    if (!he_seal_backend)
    {
        throw ngraph_error("HE backend not seal type");
    }
    float x, y;
    he_seal_backend->decode(&x, arg0, type);
    he_seal_backend->decode(&y, arg1, type);
    float r = x + y;
    he_seal_backend->encode(out, &r, type);
}
