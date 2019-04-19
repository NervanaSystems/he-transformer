//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>
#include <vector>

#include "he_backend.hpp"
#include "kernel/add.hpp"
#include "kernel/multiply.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/convolution_seal.hpp"

namespace ngraph {
namespace runtime {
namespace he {
namespace kernel {
template <typename S, typename T, typename V>
void convolution(const std::vector<std::shared_ptr<S>>& arg0,
                 const std::vector<std::shared_ptr<T>>& arg1,
                 std::vector<std::shared_ptr<V>>& out, const Shape& arg0_shape,
                 const Shape& arg1_shape, const Shape& out_shape,
                 const Strides& window_movement_strides,
                 const Strides& window_dilation_strides,
                 const CoordinateDiff& padding_below,
                 const CoordinateDiff& padding_above,
                 const Strides& data_dilation_strides, size_t batch_axis_data,
                 size_t input_channel_axis_data,
                 size_t input_channel_axis_filters,
                 size_t output_channel_axis_filters, size_t batch_axis_result,
                 size_t output_channel_axis_result, bool rotate_filter,
                 const element::Type& element_type, size_t batch_size,
                 runtime::he::HEBackend* he_backend);
}
}  // namespace he
}  // namespace runtime
}  // namespace ngraph

template <typename S, typename T, typename V>
void ngraph::runtime::he::kernel::convolution(
    const std::vector<std::shared_ptr<S>>& arg0,
    const std::vector<std::shared_ptr<T>>& arg1,
    std::vector<std::shared_ptr<V>>& out, const Shape& arg0_shape,
    const Shape& arg1_shape, const Shape& out_shape,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides, const CoordinateDiff& padding_below,
    const CoordinateDiff& padding_above, const Strides& data_dilation_strides,
    size_t batch_axis_data, size_t input_channel_axis_data,
    size_t input_channel_axis_filters, size_t output_channel_axis_filters,
    size_t batch_axis_result, size_t output_channel_axis_result,
    bool rotate_filter, const element::Type& element_type, size_t batch_size,
    runtime::he::HEBackend* he_backend) {
  // Use optimized SEAL conv if possible
  if (auto he_seal_backend =
          dynamic_cast<runtime::he::he_seal::HESealBackend*>(he_backend)) {
    runtime::he::he_seal::kernel::convolution_seal(
        arg0, arg1, out, arg0_shape, arg1_shape, out_shape,
        window_movement_strides, window_dilation_strides, padding_below,
        padding_above, data_dilation_strides, batch_axis_data,
        input_channel_axis_data, input_channel_axis_filters,
        output_channel_axis_filters, batch_axis_result,
        output_channel_axis_result, rotate_filter, element_type, batch_size,
        he_seal_backend);
    return;
  } else {
    throw ngraph_error(
        "Non-seal backend doesn't have convolution implementation");
  }
}
