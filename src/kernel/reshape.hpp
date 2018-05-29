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

#pragma once

#include <memory>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            class HEBackend;
            class HECiphertext;
            class HEPlaintext;

            namespace kernel
            {
                void reshape(const std::vector<std::shared_ptr<HECiphertext>>& arg,
                             std::vector<std::shared_ptr<he::HECiphertext>>& out,
                             const Shape& in_shape,
                             const AxisVector& in_axis_order,
                             const Shape& out_shape);

                void reshape(const std::vector<std::shared_ptr<he::HEPlaintext>>& arg,
                             std::vector<std::shared_ptr<he::HEPlaintext>>& out,
                             const Shape& in_shape,
                             const AxisVector& in_axis_order,
                             const Shape& out_shape);

                void reshape(const std::vector<std::shared_ptr<he::HEPlaintext>>& arg0,
                             std::vector<std::shared_ptr<he::HECiphertext>>& out,
                             const Shape& in_shape,
                             const AxisVector& in_axis_order,
                             const Shape& out_shape,
                             std::shared_ptr<HEBackend> he_backend);
            }
        }
    }
}
