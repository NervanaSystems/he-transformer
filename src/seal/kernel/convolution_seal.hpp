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

#include "logging/ngraph_he_log.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/add_seal.hpp"
#include "seal/kernel/multiply_seal.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph::he {

void convolution_seal(
    const std::vector<HEType>& arg0, const std::vector<HEType>& arg1,
    std::vector<HEType>& out, const Shape& arg0_shape, const Shape& arg1_shape,
    const Shape& out_shape, const Strides& window_movement_strides,
    const Strides& window_dilation_strides, const CoordinateDiff& padding_below,
    const CoordinateDiff& padding_above, const Strides& data_dilation_strides,
    size_t batch_axis_data, size_t input_channel_axis_data,
    size_t input_channel_axis_filters, size_t output_channel_axis_filters,
    size_t batch_axis_result, size_t output_channel_axis_result,
    bool rotate_filter, const element::Type& element_type, size_t batch_size,
    HESealBackend& he_seal_backend, bool verbose = true);

}  // namespace ngraph::he
