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

#include "he_backend.hpp"

using namespace ngraph;
using namespace std;

runtime::he::HEBackend::HEBackend()
{
}

runtime::he::HEBackend::~HEBackend()
{
}

shared_ptr<runtime::TensorView>
    runtime::he::HEBackend::create_tensor(const element::Type& element_type, const Shape& shape)
{
    throw ngraph_error("Unimplemented");
}

shared_ptr<runtime::TensorView> runtime::he::HEBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    throw ngraph_error("Unimplemented");
}

bool runtime::he::HEBackend::compile(std::shared_ptr<Function> func)
{
    throw ngraph_error("Unimplemented");
}

bool runtime::he::HEBackend::call(std::shared_ptr<Function> func,
                                  const vector<shared_ptr<runtime::TensorView>>& outputs,
                                  const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    throw ngraph_error("Unimplemented");
}

void runtime::he::HEBackend::remove_compiled_function(std::shared_ptr<Function> func)
{
    throw ngraph_error("Unimplemented");
}
