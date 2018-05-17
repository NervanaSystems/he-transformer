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

void runtime::he::kernel::add(const vector<shared_ptr<seal::Ciphertext>>& arg0,
                              const vector<shared_ptr<seal::Ciphertext>>& arg1,
                              vector<shared_ptr<seal::Ciphertext>>& out,
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
        he_seal_backend->get_evaluator()->add(*arg0[i], *arg1[i], *out[i]);
    }
}

void runtime::he::kernel::add(const vector<shared_ptr<seal::Ciphertext>>& arg0,
                              const vector<shared_ptr<seal::Plaintext>>& arg1,
                              vector<shared_ptr<seal::Ciphertext>>& out,
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
        he_seal_backend->get_evaluator()->add_plain(*arg0[i], *arg1[i], *out[i]);
    }
}

void runtime::he::kernel::add(const vector<shared_ptr<seal::Plaintext>>& arg0,
                              const vector<shared_ptr<seal::Ciphertext>>& arg1,
                              vector<shared_ptr<seal::Ciphertext>>& out,
                              shared_ptr<HEBackend> he_backend,
                              size_t count)
{
    add(arg1, arg0, out, he_backend, count);
}

void runtime::he::kernel::add(const vector<shared_ptr<seal::Plaintext>>& arg0,
                              const vector<shared_ptr<seal::Plaintext>>& arg1,
                              vector<shared_ptr<seal::Plaintext>>& out,
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
        auto evaluator = he_seal_backend->get_evaluator();
        float x, y;
        he_backend->decode(&x, *arg0[i], type);
        he_backend->decode(&y, *arg1[i], type);
        float r = x + y;
        he_backend->encode(*out[i], &r, type);
    }
}

void runtime::he::kernel::scalar_add(const shared_ptr<seal::Ciphertext>& arg0,
                                     const shared_ptr<seal::Ciphertext>& arg1,
                                     shared_ptr<seal::Ciphertext>& out,
                                     shared_ptr<HEBackend> he_backend)
{
	auto he_seal_backend = dynamic_pointer_cast<HESealBackend>(he_backend);
	if (!he_seal_backend)
	{
		throw ngraph_error("HE backend not seal type");
	}
    he_seal_backend->get_evaluator()->add(*arg0, *arg1, *out);
}

void runtime::he::kernel::scalar_add(const shared_ptr<seal::Plaintext>& arg0,
                                     const shared_ptr<seal::Plaintext>& arg1,
                                     shared_ptr<seal::Plaintext>& out,
                                     const element::Type& type,
                                     shared_ptr<HEBackend> he_backend)
{
    float x, y;
    he_backend->decode(&x, *arg0, type);
    he_backend->decode(&y, *arg1, type);
    float r = x + y;
    he_backend->encode(*out, &r, type);
}
