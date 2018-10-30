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

#include <cmath>
#include <utility>

#include "kernel/reshape.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void kernel::reshape(const std::vector<std::shared_ptr<HECiphertext>>& arg,
                     std::vector<std::shared_ptr<HECiphertext>>& out,
                     const Shape& in_shape,
                     const AxisVector& in_axis_order,
                     const Shape& out_shape)
{
    kernel::reshape_template(arg, out, in_shape, in_axis_order, out_shape);
}

void kernel::reshape(const std::vector<std::shared_ptr<HEPlaintext>>& arg,
                     std::vector<std::shared_ptr<HEPlaintext>>& out,
                     const Shape& in_shape,
                     const AxisVector& in_axis_order,
                     const Shape& out_shape)
{
    kernel::reshape_template(arg, out, in_shape, in_axis_order, out_shape);
}