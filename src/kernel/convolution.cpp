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

#include <vector>

#include "ngraph/type/element_type.hpp"

#include "he_backend.hpp"
#include "he_cipher_tensor_view.hpp"
#include "he_ciphertext.hpp"
#include "he_heaan_backend.hpp"
#include "he_seal_backend.hpp"
#include "kernel/add.hpp"
#include "kernel/convolution.hpp"
#include "kernel/multiply.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::convolution(const vector<shared_ptr<runtime::he::HECiphertext>>& arg0,
                                      const vector<shared_ptr<runtime::he::HEPlaintext>>& arg1,
                                      vector<shared_ptr<runtime::he::HECiphertext>>& out,
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
                                      shared_ptr<runtime::he::HEBackend> he_backend)
{
    convolution_template(arg0,
                         arg1,
                         out,
                         arg0_shape,
                         arg1_shape,
                         out_shape,
                         window_movement_strides,
                         window_dilation_strides,
                         padding_below,
                         padding_above,
                         data_dilation_strides,
                         batch_axis_data,
                         input_channel_axis_data,
                         input_channel_axis_filters,
                         output_channel_axis_filters,
                         batch_axis_result,
                         output_channel_axis_result,
                         rotate_filter,
                         type,
                         he_backend);
}

void runtime::he::kernel::convolution(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg0,
                                      const vector<shared_ptr<runtime::he::HECiphertext>>& arg1,
                                      vector<shared_ptr<runtime::he::HECiphertext>>& out,
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
                                      shared_ptr<runtime::he::HEBackend> he_backend)
{
    convolution(arg1,
                arg0,
                out,
                arg1_shape,
                arg0_shape,
                out_shape,
                window_movement_strides,
                window_dilation_strides,
                padding_below,
                padding_above,
                data_dilation_strides,
                batch_axis_data,
                input_channel_axis_data,
                input_channel_axis_filters,
                output_channel_axis_filters,
                batch_axis_result,
                output_channel_axis_result,
                rotate_filter,
                type,
                he_backend);
}

void runtime::he::kernel::convolution(const vector<shared_ptr<runtime::he::HECiphertext>>& arg0,
                                      const vector<shared_ptr<runtime::he::HECiphertext>>& arg1,
                                      vector<shared_ptr<runtime::he::HECiphertext>>& out,
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
                                      shared_ptr<runtime::he::HEBackend> he_backend)
{
    convolution_template(arg0,
                         arg1,
                         out,
                         arg0_shape,
                         arg1_shape,
                         out_shape,
                         window_movement_strides,
                         window_dilation_strides,
                         padding_below,
                         padding_above,
                         data_dilation_strides,
                         batch_axis_data,
                         input_channel_axis_data,
                         input_channel_axis_filters,
                         output_channel_axis_filters,
                         batch_axis_result,
                         output_channel_axis_result,
                         rotate_filter,
                         type,
                         he_backend);
}
