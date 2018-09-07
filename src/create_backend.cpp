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

extern "C" const char* get_ngraph_version_string()
{
    return "v0.7.0";
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    return new runtime::he::he_seal::HESealBackend();
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}
