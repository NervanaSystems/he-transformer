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
#include "seal/seal.h"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::add(const vector<shared_ptr<seal::Ciphertext>>& arg0,
                              const vector<shared_ptr<seal::Ciphertext>>& arg1,
                              vector<shared_ptr<seal::Ciphertext>>& out,
                              shared_ptr<HEBackend> he_backend,
                              size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        he_backend.get()->get_evaluator()->add(*arg0[i], *arg1[i], *out[i]);
    }
}

void runtime::he::kernel::add(const shared_ptr<seal::Ciphertext>& arg0,
                              const shared_ptr<seal::Ciphertext>& arg1,
                              shared_ptr<seal::Ciphertext>& out,
                              shared_ptr<HEBackend> he_backend)
{
    const vector<shared_ptr<seal::Ciphertext>> arg0vec = {arg0};
    const vector<shared_ptr<seal::Ciphertext>> arg1vec = {arg1};
    vector<shared_ptr<seal::Ciphertext>> outvec = {out};
    add(arg0vec, arg1vec, outvec, he_backend, 1);
}
