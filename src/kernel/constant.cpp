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
#include "kernel/constant.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/seal.h"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::constant(vector<shared_ptr<seal::Ciphertext>>& out,
                                   const element::Type& type,
                                   shared_ptr<Node> node,
                                   shared_ptr<HEBackend> he_backend,
                                   size_t count)
{
    shared_ptr<op::Constant> constant = static_pointer_cast<op::Constant>(node);
    size_t type_byte_size = type.size();
    if (out.size() != count)
    {
        throw ngraph_error("out.size() != count for constant op");
    }
    for (size_t i = 0; i < count; ++i)
    {
        const void* src_with_offset = (void*)((char*)constant->get_data_ptr() + i * type.size());
        seal::Plaintext p;
        he_backend->encode(p, src_with_offset, type);
        he_backend->encrypt(*(out[i]), p);
    }
}
