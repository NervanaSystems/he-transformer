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

#include "he_backend.hpp"

using namespace std;
using namespace ngraph;

extern "C" bool create_backend()
{
    // This is called if compiled by GCC. The static_init() only works on clang.
    runtime::Backend::register_backend("HE", make_shared<runtime::nnp::HEBackend>());
    return true;
}
