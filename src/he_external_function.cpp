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

#include "he_external_function.hpp"
using namespace std;
using namespace ngraph;

runtime::he::HEExternalFunction::HEExternalFunction(const shared_ptr<Function>& function,
                                                    bool release_function)
    : runtime::ExternalFunction(function, release_function)
{
}

void runtime::he::HEExternalFunction::compile()
{
    throw ngraph_error("Unimplemented");
}

shared_ptr<runtime::CallFrame> runtime::he::HEExternalFunction::make_call_frame()
{
    throw ngraph_error("Unimplemented");
}
