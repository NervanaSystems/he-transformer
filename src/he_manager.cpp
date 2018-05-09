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

#include <memory>

#include "he_backend.hpp"
#include "he_manager.hpp"

using namespace ngraph;
using namespace std;

shared_ptr<runtime::Backend> runtime::he::HEManager::allocate_backend()
{
    return make_shared<HEBackend>();
}

vector<size_t> runtime::he::HEManager::get_subdevices() const
{
    throw ngraph_error("unimplemented");
}

bool REGISTER_HE_RUNTIME()
{
    runtime::Manager::register_factory("HE",
                                       [](const string& name) -> shared_ptr<runtime::Manager> {
                                           return make_shared<runtime::he::HEManager>();
                                       });
    return true;
}
