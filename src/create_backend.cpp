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

#include <unordered_map>

#include "ngraph/runtime/backend.hpp"

#include "he_heaan_backend.hpp"
#include "he_seal_backend.hpp"

using namespace std;
using namespace ngraph;

// extern "C" bool create_backend()
// {
//     NGRAPH_INFO << "Create_backend";
//     runtime::Backend::register_backend("HE_HEAAN",
//                                        make_shared<runtime::he::he_heaan::HEHeaanBackend>());
//     runtime::Backend::register_backend("HE_SEAL",
//                                        make_shared<runtime::he::he_seal::HESealBackend>());
//     return true;
// }

// Hack to avoid weak pointer error
// TODO: the best solution is to remove all `shared_from_this()` from the code
static unordered_map<runtime::Backend*, shared_ptr<runtime::Backend>>
    s_map_backend_ptr_to_shared_ptr;

extern "C" const char* get_ngraph_version_string()
{
    return "v0.7.0";
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    shared_ptr<runtime::Backend> he_backend;
    if (string(configuration_string) == "HE:SEAL")
    {
        he_backend = make_shared<runtime::he::he_seal::HESealBackend>();
    }
    else if (string(configuration_string) == "HE:HEAAN")
    {
        he_backend = make_shared<runtime::he::he_heaan::HEHeaanBackend>();
    }
    s_map_backend_ptr_to_shared_ptr[he_backend.get()] = he_backend;
    return he_backend.get();
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    s_map_backend_ptr_to_shared_ptr.erase(backend);
}
