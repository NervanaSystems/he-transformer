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

#include <stack>
#include <vector>

#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "he_tensor_view.hpp"

using namespace std;
using namespace ngraph;

runtime::he::HETensorView::~HETensorView()
{
}

void runtime::he::HETensorView::write(const void* p, size_t tensor_offset, size_t n)
{
    throw ngraph_error("not implemented");
}

void runtime::he::HETensorView::read(void* p, size_t tensor_offset, size_t n) const
{
    throw ngraph_error("not implemented");
}
