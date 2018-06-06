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

#include "kernel/constant.hpp"
#include "he_heaan_backend.hpp"
#include "he_seal_backend.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::constant(vector<shared_ptr<runtime::he::HEPlaintext>>& out,
                                   const element::Type& type,
                                   const void* data_ptr,
                                   const shared_ptr<runtime::he::HEBackend>& he_backend,
                                   size_t count)
{
    size_t type_byte_size = type.size();
    if (out.size() != count)
    {
        throw ngraph_error("out.size() != count for constant op");
    }

    if (auto he_seal_backend =
            dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(he_backend))
    {
#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            const void* src_with_offset = (void*)((char*)data_ptr + i * type.size());
            he_seal_backend->encode(out[i], src_with_offset, type);
        }
    }
    else if (auto he_heaan_backend =
                 dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(he_backend))
    {
#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            const void* src_with_offset = (void*)((char*)data_ptr + i * type.size());
            he_heaan_backend->encode(out[i], src_with_offset, type);
        }
    }
    else
    {
        throw ngraph_error("Constant backend is neither seal nor hean.");
    }
}
