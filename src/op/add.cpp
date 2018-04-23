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

#include "he_cipher_tensor_view.hpp"
#include "he_backend.hpp"
#include "add.hpp"

void ngraph::runtime::he::add(const HECipherTensorView* arg0, const HECipherTensorView* arg1, HECipherTensorView* out, size_t count)
{
    for(size_t i = 0; i < count; ++i)
    {
        arg0->m_he_backend->get_evaluator()->add(arg0->get_element(i), arg1->get_element(i), out->get_element(i));
    }
}


