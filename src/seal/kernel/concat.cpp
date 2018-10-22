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
#include <vector>

#include "he_ciphertext.hpp"
#include "kernel/concat.hpp"
#include "ngraph/coordinate_transform.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::concat(const vector<vector<shared_ptr<runtime::he::HECiphertext>>>& args,
                                 vector<shared_ptr<runtime::he::HECiphertext>>& out,
                                 const vector<Shape>& in_shapes,
                                 const Shape& out_shape,
                                 size_t concatenation_axis)
{
    concat<runtime::he::HECiphertext, runtime::he::HECiphertext>(
        args, out, in_shapes, out_shape, concatenation_axis);
}

void runtime::he::kernel::concat(const vector<vector<shared_ptr<runtime::he::HEPlaintext>>>& args,
                                 vector<shared_ptr<runtime::he::HEPlaintext>>& out,
                                 const vector<Shape>& in_shapes,
                                 const Shape& out_shape,
                                 size_t concatenation_axis)
{
    concat<runtime::he::HEPlaintext, runtime::he::HEPlaintext>(
        args, out, in_shapes, out_shape, concatenation_axis);
}
