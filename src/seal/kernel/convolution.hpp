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

#include "he_seal_backend.hpp"
#include "kernel/add.hpp"
#include "kernel/multiply.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"

namespace ngraph {
namespace he {
inline void convolution(
    const std::vector<std::shared_ptr<HECiphertext>>& arg0,
    const std::vector<std::shared_ptr<HECiphertext>>& arg1,
    std::vector<std::shared_ptr<HECiphertext>>& out, const Shape& arg0_shape,
    const Shape& arg1_shape, const Shape& out_shape,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides, const CoordinateDiff& padding_below,
    const CoordinateDiff& padding_above, const Strides& data_dilation_strides,
    size_t batch_axis_data, size_t input_channel_axis_data,
    size_t input_channel_axis_filters, size_t output_channel_axis_filters,
    size_t batch_axis_result, size_t output_channel_axis_result,
    bool rotate_filter, const element::Type& element_type, size_t batch_size,
    const ngraph::he::HESealBasckend* he_seal_backend) {
  throw ngraph_error("conv unimplemented");
}

inline void convolution(
    const std::vector<std::unique_ptr<HEPlaintext>>& arg0,
    const std::vector<std::unique_ptr<HEPlaintext>>& arg1,
    std::vector<std::unique_ptr<HEPlaintext>>& out, const Shape& arg0_shape,
    const Shape& arg1_shape, const Shape& out_shape,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides, const CoordinateDiff& padding_below,
    const CoordinateDiff& padding_above, const Strides& data_dilation_strides,
    size_t batch_axis_data, size_t input_channel_axis_data,
    size_t input_channel_axis_filters, size_t output_channel_axis_filters,
    size_t batch_axis_result, size_t output_channel_axis_result,
    bool rotate_filter, const element::Type& element_type, size_t batch_size,
    const ngraph::he::HESealBasckend* he_seal_backend) {
  throw ngraph_error("conv unimplemented");
}

inline void convolution(
    const std::vector<std::unique_ptr<HEPlaintext>>& arg0,
    const std::vector<std::shared_ptr<HECiphertext>>& arg1,
    std::vector<std::shared_ptr<HECiphertext>>& out, const Shape& arg0_shape,
    const Shape& arg1_shape, const Shape& out_shape,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides, const CoordinateDiff& padding_below,
    const CoordinateDiff& padding_above, const Strides& data_dilation_strides,
    size_t batch_axis_data, size_t input_channel_axis_data,
    size_t input_channel_axis_filters, size_t output_channel_axis_filters,
    size_t batch_axis_result, size_t output_channel_axis_result,
    bool rotate_filter, const element::Type& element_type, size_t batch_size,
    const ngraph::he::HESealBasckend* he_seal_backend) {
  throw ngraph_error("conv unimplemented");
}
inline void convolution(
    const std::vector<std::shared_ptr<HECiphertext>>& arg0,
    const std::vector<std::unique_ptr<HEPlaintext>>& arg1,
    std::vector<std::shared_ptr<HECiphertext>>& out, const Shape& arg0_shape,
    const Shape& arg1_shape, const Shape& out_shape,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides, const CoordinateDiff& padding_below,
    const CoordinateDiff& padding_above, const Strides& data_dilation_strides,
    size_t batch_axis_data, size_t input_channel_axis_data,
    size_t input_channel_axis_filters, size_t output_channel_axis_filters,
    size_t batch_axis_result, size_t output_channel_axis_result,
    bool rotate_filter, const element::Type& element_type, size_t batch_size,
    const ngraph::he::HESealBasckend* he_seal_backend) {
  throw ngraph_error("conv unimplemented");
}
}  // namespace he
}  // namespace ngraph
