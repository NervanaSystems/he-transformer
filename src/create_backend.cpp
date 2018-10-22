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

#include "he_ckks_backend.hpp"
#include "he_seal_backend.hpp"

using namespace std;
using namespace ngraph;

// extern "C" bool create_backend()
// {
//     NGRAPH_INFO << "Create_backend";
//     runtime::Backend::register_backend("HE_HEAAN",
//                                        make_shared<runtime::he::he_ckks::HEHeaanBackend>());
//     runtime::Backend::register_backend("HE_SEAL",
//                                        make_shared<runtime::he::he_seal::HESealBackend>());
//     return true;
// }

// Hack to avoid weak pointer error
// TODO: the best solution is to remove all `shared_from_this()` from the code
static unordered_map<string, shared_ptr<runtime::Backend>> s_map_backends;

extern "C" const char* get_ngraph_version_string()
{
    return "v0.7.0";
}

extern "C" runtime::Backend* new_backend(const char* configuration_chars)
{
    string configuration_string = string(configuration_chars);
    if (s_map_backends.find(configuration_string) == s_map_backends.end())
    {
        shared_ptr<runtime::Backend> he_backend;
        if (configuration_string == "HE:SEAL")
        {
            he_backend = make_shared<runtime::he::he_seal::HESealBackend>();
        }
        else if (configuration_string == "HE:HEAAN")
        {
            he_backend = make_shared<runtime::he::he_ckks::HEHeaanBackend>();
        }
        s_map_backends[configuration_string] = he_backend;
    }

    return s_map_backends.at(configuration_string).get();
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    // s_map_backend_ptr_to_shared_ptr.erase(backend);
}
