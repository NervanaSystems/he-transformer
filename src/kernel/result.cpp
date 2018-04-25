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

#include "kernel/result.hpp"
#include "seal/seal.h"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::result(const vector<shared_ptr<seal::Ciphertext>> arg,
                                 vector<shared_ptr<seal::Ciphertext>>& out,
                                 size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        out[i] = arg[i];
    }
}

void runtime::he::kernel::result(const vector<shared_ptr<seal::Plaintext>> arg,
        vector<shared_ptr<seal::Plaintext>>& out,
        size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        out[i] = arg[i];
    }
}
