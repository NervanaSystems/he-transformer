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

#include <vector>
#include <memory>

#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace element
    {
        class Type;
    }
    namespace runtime
    {
        namespace he
        {
            class HEBackend;
            class HECiphertext;
            namespace kernel
            {
                void convolution(
                        const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& arg0,
                        const std::vector<std::shared_ptr<runtime::he::HEPlaintext>>& arg1,
                         std::vector<std::shared_ptr<runtime::he::HECiphertext>>& out,
                         const Shape& arg0_shape,
                         const Shape& arg1_shape,
                         const Shape& out_shape,
                         const Strides& window_movement_strides,
                         const Strides& window_dilation_strides,
                         const CoordinateDiff& padding_below,
                         const CoordinateDiff& padding_above,
                         const Strides& data_dilation_strides,
                         size_t batch_axis_data,
                         size_t input_channel_axis_data,
                         size_t input_channel_axis_filters,
                         size_t output_channel_axis_filters,
                         size_t batch_axis_result,
                         size_t output_channel_axis_result,
                         bool rotate_filter,
                         const element::Type& type,
                         std::shared_ptr<runtime::he::HEBackend> he_backend);
            }
        }
    }
}
