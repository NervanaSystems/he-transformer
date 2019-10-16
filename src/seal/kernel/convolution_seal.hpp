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
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace he {
inline void convolution_seal(
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg0,
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const Shape& arg0_shape, const Shape& arg1_shape, const Shape& out_shape,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides, const CoordinateDiff& padding_below,
    const CoordinateDiff& padding_above, const Strides& data_dilation_strides,
    size_t batch_axis_data, size_t input_channel_axis_data,
    size_t input_channel_axis_filters, size_t output_channel_axis_filters,
    size_t batch_axis_result, size_t output_channel_axis_result,
    bool rotate_filter, const element::Type& element_type, size_t batch_size,
    HESealBackend& he_seal_backend, bool verbose = true) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
  // Comments throughout assume without loss of generality that:
  //
  // * batch axes for both input data and output data are 0
  // * input channel axes for both input data and filters are 1
  // * output channel axes for filters is 0
  // * output channel axis for output data is 1
  // * rotate_filter is false

  // At the outermost level we will walk over every output coordinate O.
  CoordinateTransform output_transform(out_shape);

  // Store output coordinates for parallelization
  std::vector<ngraph::Coordinate> out_coords;
  for (const Coordinate& out_coord : output_transform) {
    out_coords.emplace_back(out_coord);
  }
  size_t out_transform_size = out_coords.size();
  if (verbose) {
    NGRAPH_HE_LOG(5) << "Convolution output size " << out_transform_size;
  }

#pragma omp parallel for
  for (size_t out_coord_idx = 0; out_coord_idx < out_transform_size;
       ++out_coord_idx) {
    // Init thread-local memory pool for each thread
    seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::ThreadLocal();

    const Coordinate& out_coord = out_coords[out_coord_idx];

    // for (Coordinate out_coord : output_transform)
    //{
    // Our output coordinate O will have the form:
    //
    //   (N,chan_out,i_1,...,i_n)

    size_t batch_index = out_coord[batch_axis_result];
    size_t output_channel = out_coord[output_channel_axis_result];

    // For the input data we need to iterate the coordinate:
    //
    //   I:
    //
    // over the range (noninclusive on the right):
    //
    //   (N,0,s_1*i_1,s_2*i_2,...,s_n*i_n) ->
    //
    //     (N+1,chans_in_count,s_1*i_1 + l_1*filter_dims_1,...,s_n*i_n +
    //     l_n*filter_dims_n)
    //
    // with strides:
    //
    //   (1,l_1,...,l_n).
    //
    // Note that we are iterating within the *padded* and *dilated* data batch,
    // so further
    // down we must check the current coordinate is in the padding or dilation
    // gap.

    size_t n_spatial_dimensions = arg0_shape.size() - 2;
    size_t n_input_channels = arg0_shape[input_channel_axis_data];

    Coordinate input_batch_transform_start(2 + n_spatial_dimensions);
    Coordinate input_batch_transform_end(2 + n_spatial_dimensions);
    Strides input_batch_transform_movement_strides(2 + n_spatial_dimensions, 1);
    CoordinateDiff input_batch_transform_padding_below(2 + n_spatial_dimensions,
                                                       0);
    CoordinateDiff input_batch_transform_padding_above(2 + n_spatial_dimensions,
                                                       0);
    Strides input_batch_transform_dilation_strides(2 + n_spatial_dimensions, 1);

    input_batch_transform_start[batch_axis_data] = batch_index;
    input_batch_transform_end[batch_axis_data] = batch_index + 1;
    input_batch_transform_start[input_channel_axis_data] = 0;
    input_batch_transform_end[input_channel_axis_data] = n_input_channels;

    for (size_t i = 2; i < n_spatial_dimensions + 2; i++) {
      size_t window_dilation_stride = window_dilation_strides[i - 2];
      size_t window_movement_stride = window_movement_strides[i - 2];
      std::ptrdiff_t below_pad = padding_below[i - 2];
      std::ptrdiff_t above_pad = padding_above[i - 2];
      size_t data_dilation_stride = data_dilation_strides[i - 2];

      input_batch_transform_start[i] = window_movement_stride * out_coord[i];
      input_batch_transform_end[i] =
          input_batch_transform_start[i] +
          (arg1_shape[i] - 1) * window_dilation_stride + 1;
      input_batch_transform_movement_strides[i] = window_dilation_stride;
      input_batch_transform_padding_below[i] = below_pad;
      input_batch_transform_padding_above[i] = above_pad;
      input_batch_transform_dilation_strides[i] = data_dilation_stride;
    }

    AxisVector input_batch_transform_axis_order(2 + n_spatial_dimensions);
    for (size_t i = 0; i < input_batch_transform_axis_order.size(); i++) {
      input_batch_transform_axis_order[i] = i;
    }

    CoordinateTransform input_batch_transform(
        arg0_shape, input_batch_transform_start, input_batch_transform_end,
        input_batch_transform_movement_strides,
        input_batch_transform_axis_order, input_batch_transform_padding_below,
        input_batch_transform_padding_above,
        input_batch_transform_dilation_strides);

    // Simultaneously with iterating I, for the filters we need to iterate the
    // coordinate:
    //
    //   F
    //
    // over the range (noninclusive on the right):
    //
    //   (chan_out,0,0,...,0) ->
    //   (chan_out+1,chans_in_count,filter_dims_1,...,filter_dims_n)
    //
    // with unit stride.

    Shape filter_transform_start(2 + n_spatial_dimensions);
    Shape filter_transform_end(2 + n_spatial_dimensions);

    filter_transform_start[output_channel_axis_filters] = output_channel;
    filter_transform_end[output_channel_axis_filters] = output_channel + 1;
    filter_transform_start[input_channel_axis_filters] = 0;
    filter_transform_end[input_channel_axis_filters] = n_input_channels;

    for (size_t i = 2; i < n_spatial_dimensions + 2; i++) {
      filter_transform_start[i] = 0;
      filter_transform_end[i] = arg1_shape[i];
    }

    CoordinateTransform filter_transform(arg1_shape, filter_transform_start,
                                         filter_transform_end);

    // As we go, we sum up:
    //
    //   output[O] += arg0[I] * arg1[F].

    // T result = 0;

    CoordinateTransform::Iterator input_it = input_batch_transform.begin();
    CoordinateTransform::Iterator filter_it = filter_transform.begin();
    CoordinateTransform::Iterator input_end = input_batch_transform.end();
    CoordinateTransform::Iterator filter_end = filter_transform.end();

    auto sum = he_seal_backend.create_empty_ciphertext(pool);
    bool first_add = true;

    while (input_it != input_end && filter_it != filter_end) {
      const Coordinate& input_batch_coord = *input_it;
      Coordinate filter_coord = *filter_it;

      if (rotate_filter) {
        Shape target_shape = filter_transform.get_target_shape();

        // Note that we only reverse the spatial dimensions here (loop
        // starts at 2)
        for (size_t i = 2; i < filter_coord.size(); i++) {
          filter_coord[i] = target_shape[i] - filter_coord[i] - 1;
        }
      }

      if (input_batch_transform.has_source_coordinate(input_batch_coord)) {
        auto mult_arg0 = arg0[input_batch_transform.index(input_batch_coord)];

        auto mult_arg1 = arg1[filter_transform.index(filter_coord)];
        auto prod = he_seal_backend.create_empty_ciphertext(pool);

        scalar_multiply_seal(*mult_arg0, *mult_arg1, prod, element_type,
                             he_seal_backend, pool);
        if (first_add) {
          sum = prod;
          first_add = false;
        } else {
          scalar_add_seal(*prod, *sum, sum, element_type, he_seal_backend,
                          pool);
        }
      }
      ++input_it;
      ++filter_it;
    }
    if (first_add) {
      out[out_coord_idx] = std::make_shared<SealCiphertextWrapper>();
      out[out_coord_idx]->known_value() = true;
      out[out_coord_idx]->value() = 0;
    } else {
      // Write the sum back.
      out[out_coord_idx] = sum;
    }

    if (verbose && out_coord_idx % 1000 == 0 && out_coord_idx != 0) {
      NGRAPH_HE_LOG(3) << "Finished out coord " << out_coord_idx;
    }
  }
}

inline void convolution_seal(
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg0,
    const std::vector<HEPlaintext>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const Shape& arg0_shape, const Shape& arg1_shape, const Shape& out_shape,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides, const CoordinateDiff& padding_below,
    const CoordinateDiff& padding_above, const Strides& data_dilation_strides,
    size_t batch_axis_data, size_t input_channel_axis_data,
    size_t input_channel_axis_filters, size_t output_channel_axis_filters,
    size_t batch_axis_result, size_t output_channel_axis_result,
    bool rotate_filter, const element::Type& element_type, size_t batch_size,
    HESealBackend& he_seal_backend, bool verbose = true) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type", element_type);
  CoordinateTransform output_transform(out_shape);

  // Store output coordinates for parallelization
  std::vector<ngraph::Coordinate> out_coords;
  for (const Coordinate& out_coord : output_transform) {
    out_coords.emplace_back(out_coord);
  }
  size_t out_transform_size = out_coords.size();
  if (verbose) {
    NGRAPH_HE_LOG(3) << "Convolution output size " << out_transform_size;
  }

#pragma omp parallel for
  for (size_t out_coord_idx = 0; out_coord_idx < out_transform_size;
       ++out_coord_idx) {
    // Init thread-local memory pool for each thread
    seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::ThreadLocal();

    const Coordinate& out_coord = out_coords[out_coord_idx];

    size_t batch_index = out_coord[batch_axis_result];
    size_t output_channel = out_coord[output_channel_axis_result];

    size_t n_spatial_dimensions = arg0_shape.size() - 2;
    size_t n_input_channels = arg0_shape[input_channel_axis_data];

    Coordinate input_batch_transform_start(2 + n_spatial_dimensions);
    Coordinate input_batch_transform_end(2 + n_spatial_dimensions);
    Strides input_batch_transform_movement_strides(2 + n_spatial_dimensions, 1);
    CoordinateDiff input_batch_transform_padding_below(2 + n_spatial_dimensions,
                                                       0);
    CoordinateDiff input_batch_transform_padding_above(2 + n_spatial_dimensions,
                                                       0);
    Strides input_batch_transform_dilation_strides(2 + n_spatial_dimensions, 1);

    input_batch_transform_start[batch_axis_data] = batch_index;
    input_batch_transform_end[batch_axis_data] = batch_index + 1;
    input_batch_transform_start[input_channel_axis_data] = 0;
    input_batch_transform_end[input_channel_axis_data] = n_input_channels;

    for (size_t i = 2; i < n_spatial_dimensions + 2; i++) {
      size_t window_dilation_stride = window_dilation_strides[i - 2];
      size_t window_movement_stride = window_movement_strides[i - 2];
      std::ptrdiff_t below_pad = padding_below[i - 2];
      std::ptrdiff_t above_pad = padding_above[i - 2];
      size_t data_dilation_stride = data_dilation_strides[i - 2];

      input_batch_transform_start[i] = window_movement_stride * out_coord[i];
      input_batch_transform_end[i] =
          input_batch_transform_start[i] +
          (arg1_shape[i] - 1) * window_dilation_stride + 1;
      input_batch_transform_movement_strides[i] = window_dilation_stride;
      input_batch_transform_padding_below[i] = below_pad;
      input_batch_transform_padding_above[i] = above_pad;
      input_batch_transform_dilation_strides[i] = data_dilation_stride;
    }

    AxisVector input_batch_transform_axis_order(2 + n_spatial_dimensions);
    for (size_t i = 0; i < input_batch_transform_axis_order.size(); i++) {
      input_batch_transform_axis_order[i] = i;
    }

    CoordinateTransform input_batch_transform(
        arg0_shape, input_batch_transform_start, input_batch_transform_end,
        input_batch_transform_movement_strides,
        input_batch_transform_axis_order, input_batch_transform_padding_below,
        input_batch_transform_padding_above,
        input_batch_transform_dilation_strides);

    Shape filter_transform_start(2 + n_spatial_dimensions);
    Shape filter_transform_end(2 + n_spatial_dimensions);

    filter_transform_start[output_channel_axis_filters] = output_channel;
    filter_transform_end[output_channel_axis_filters] = output_channel + 1;
    filter_transform_start[input_channel_axis_filters] = 0;
    filter_transform_end[input_channel_axis_filters] = n_input_channels;

    for (size_t i = 2; i < n_spatial_dimensions + 2; i++) {
      filter_transform_start[i] = 0;
      filter_transform_end[i] = arg1_shape[i];
    }

    CoordinateTransform filter_transform(arg1_shape, filter_transform_start,
                                         filter_transform_end);

    CoordinateTransform::Iterator input_it = input_batch_transform.begin();
    CoordinateTransform::Iterator filter_it = filter_transform.begin();
    CoordinateTransform::Iterator input_end = input_batch_transform.end();
    CoordinateTransform::Iterator filter_end = filter_transform.end();

    std::shared_ptr<SealCiphertextWrapper> sum;
    bool first_add = true;

    while (input_it != input_end && filter_it != filter_end) {
      const Coordinate& input_batch_coord = *input_it;
      Coordinate filter_coord = *filter_it;

      if (rotate_filter) {
        Shape target_shape = filter_transform.get_target_shape();
        for (size_t i = 2; i < filter_coord.size(); i++) {
          filter_coord[i] = target_shape[i] - filter_coord[i] - 1;
        }
      }

      if (input_batch_transform.has_source_coordinate(input_batch_coord)) {
        auto mult_arg0 = arg0[input_batch_transform.index(input_batch_coord)];
        auto mult_arg1 = arg1[filter_transform.index(filter_coord)];
        auto prod = he_seal_backend.create_empty_ciphertext(pool);

        scalar_multiply_seal(*mult_arg0, mult_arg1, prod, element_type,
                             he_seal_backend, pool);
        if (first_add) {
          sum = prod;
          first_add = false;
        } else {
          scalar_add_seal(*prod, *sum, sum, element_type, he_seal_backend,
                          pool);
        }
      }
      ++input_it;
      ++filter_it;
    }
    if (first_add) {
      out[out_coord_idx] = std::make_shared<SealCiphertextWrapper>();
      out[out_coord_idx]->known_value() = true;
      out[out_coord_idx]->value() = 0;
    } else {
      // Write the sum back.
      out[out_coord_idx] = sum;
    }

    if (verbose && out_coord_idx % 1000 == 0 && out_coord_idx != 0) {
      NGRAPH_HE_LOG(3) << "Finished out coord " << out_coord_idx;
    }
  }
}

inline void convolution_seal(
    const std::vector<HEPlaintext>& arg0,
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const Shape& arg0_shape, const Shape& arg1_shape, const Shape& out_shape,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides, const CoordinateDiff& padding_below,
    const CoordinateDiff& padding_above, const Strides& data_dilation_strides,
    size_t batch_axis_data, size_t input_channel_axis_data,
    size_t input_channel_axis_filters, size_t output_channel_axis_filters,
    size_t batch_axis_result, size_t output_channel_axis_result,
    bool rotate_filter, const element::Type& element_type, size_t batch_size,
    HESealBackend& he_seal_backend, bool verbose = true) {
  CoordinateTransform output_transform(out_shape);

  // Store output coordinates for parallelization
  std::vector<ngraph::Coordinate> out_coords;
  for (const Coordinate& out_coord : output_transform) {
    out_coords.emplace_back(out_coord);
  }
  size_t out_transform_size = out_coords.size();
  if (verbose) {
    NGRAPH_HE_LOG(3) << "Convolution output size " << out_transform_size;
  }

#pragma omp parallel for
  for (size_t out_coord_idx = 0; out_coord_idx < out_transform_size;
       ++out_coord_idx) {
    // Init thread-local memory pool for each thread
    seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::ThreadLocal();

    const Coordinate& out_coord = out_coords[out_coord_idx];

    size_t batch_index = out_coord[batch_axis_result];
    size_t output_channel = out_coord[output_channel_axis_result];

    size_t n_spatial_dimensions = arg0_shape.size() - 2;
    size_t n_input_channels = arg0_shape[input_channel_axis_data];

    Coordinate input_batch_transform_start(2 + n_spatial_dimensions);
    Coordinate input_batch_transform_end(2 + n_spatial_dimensions);
    Strides input_batch_transform_movement_strides(2 + n_spatial_dimensions, 1);
    CoordinateDiff input_batch_transform_padding_below(2 + n_spatial_dimensions,
                                                       0);
    CoordinateDiff input_batch_transform_padding_above(2 + n_spatial_dimensions,
                                                       0);
    Strides input_batch_transform_dilation_strides(2 + n_spatial_dimensions, 1);

    input_batch_transform_start[batch_axis_data] = batch_index;
    input_batch_transform_end[batch_axis_data] = batch_index + 1;
    input_batch_transform_start[input_channel_axis_data] = 0;
    input_batch_transform_end[input_channel_axis_data] = n_input_channels;

    for (size_t i = 2; i < n_spatial_dimensions + 2; i++) {
      size_t window_dilation_stride = window_dilation_strides[i - 2];
      size_t window_movement_stride = window_movement_strides[i - 2];
      std::ptrdiff_t below_pad = padding_below[i - 2];
      std::ptrdiff_t above_pad = padding_above[i - 2];
      size_t data_dilation_stride = data_dilation_strides[i - 2];

      input_batch_transform_start[i] = window_movement_stride * out_coord[i];
      input_batch_transform_end[i] =
          input_batch_transform_start[i] +
          (arg1_shape[i] - 1) * window_dilation_stride + 1;
      input_batch_transform_movement_strides[i] = window_dilation_stride;
      input_batch_transform_padding_below[i] = below_pad;
      input_batch_transform_padding_above[i] = above_pad;
      input_batch_transform_dilation_strides[i] = data_dilation_stride;
    }

    AxisVector input_batch_transform_axis_order(2 + n_spatial_dimensions);
    for (size_t i = 0; i < input_batch_transform_axis_order.size(); i++) {
      input_batch_transform_axis_order[i] = i;
    }

    CoordinateTransform input_batch_transform(
        arg0_shape, input_batch_transform_start, input_batch_transform_end,
        input_batch_transform_movement_strides,
        input_batch_transform_axis_order, input_batch_transform_padding_below,
        input_batch_transform_padding_above,
        input_batch_transform_dilation_strides);

    Shape filter_transform_start(2 + n_spatial_dimensions);
    Shape filter_transform_end(2 + n_spatial_dimensions);

    filter_transform_start[output_channel_axis_filters] = output_channel;
    filter_transform_end[output_channel_axis_filters] = output_channel + 1;
    filter_transform_start[input_channel_axis_filters] = 0;
    filter_transform_end[input_channel_axis_filters] = n_input_channels;

    for (size_t i = 2; i < n_spatial_dimensions + 2; i++) {
      filter_transform_start[i] = 0;
      filter_transform_end[i] = arg1_shape[i];
    }

    CoordinateTransform filter_transform(arg1_shape, filter_transform_start,
                                         filter_transform_end);

    CoordinateTransform::Iterator input_it = input_batch_transform.begin();
    CoordinateTransform::Iterator filter_it = filter_transform.begin();
    CoordinateTransform::Iterator input_end = input_batch_transform.end();
    CoordinateTransform::Iterator filter_end = filter_transform.end();

    auto sum = he_seal_backend.create_empty_ciphertext(pool);
    bool first_add = true;

    while (input_it != input_end && filter_it != filter_end) {
      const Coordinate& input_batch_coord = *input_it;
      Coordinate filter_coord = *filter_it;

      if (rotate_filter) {
        Shape target_shape = filter_transform.get_target_shape();
        for (size_t i = 2; i < filter_coord.size(); i++) {
          filter_coord[i] = target_shape[i] - filter_coord[i] - 1;
        }
      }

      if (input_batch_transform.has_source_coordinate(input_batch_coord)) {
        auto mult_arg0 = arg0[input_batch_transform.index(input_batch_coord)];

        auto mult_arg1 = arg1[filter_transform.index(filter_coord)];
        auto prod = he_seal_backend.create_empty_ciphertext(pool);

        scalar_multiply_seal(*mult_arg1, mult_arg0, prod, element_type,
                             he_seal_backend, pool);
        if (first_add) {
          sum = prod;
          first_add = false;
        } else {
          scalar_add_seal(*prod, *sum, sum, element_type, he_seal_backend,
                          pool);
        }
      }
      ++input_it;
      ++filter_it;
    }
    if (first_add) {
      out[out_coord_idx] = std::make_shared<SealCiphertextWrapper>();
      out[out_coord_idx]->known_value() = true;
      out[out_coord_idx]->value() = 0;
    } else {
      // Write the sum back.
      out[out_coord_idx] = sum;
    }

    if (verbose && out_coord_idx % 1000 == 0 && out_coord_idx != 0) {
      NGRAPH_HE_LOG(3) << "Finished out coord " << out_coord_idx;
    }
  }
}

inline void convolution_seal(
    const std::vector<HEPlaintext>& arg0, const std::vector<HEPlaintext>& arg1,
    std::vector<HEPlaintext>& out, const Shape& arg0_shape,
    const Shape& arg1_shape, const Shape& out_shape,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides, const CoordinateDiff& padding_below,
    const CoordinateDiff& padding_above, const Strides& data_dilation_strides,
    size_t batch_axis_data, size_t input_channel_axis_data,
    size_t input_channel_axis_filters, size_t output_channel_axis_filters,
    size_t batch_axis_result, size_t output_channel_axis_result,
    bool rotate_filter, const element::Type& element_type, size_t batch_size,
    HESealBackend& he_seal_backend, bool verbose = true) {
  CoordinateTransform output_transform(out_shape);

  // Store output coordinates for parallelization
  std::vector<ngraph::Coordinate> out_coords;
  for (const Coordinate& out_coord : output_transform) {
    out_coords.emplace_back(out_coord);
  }
  size_t out_transform_size = out_coords.size();
  if (verbose) {
    NGRAPH_HE_LOG(3) << "Convolution output size " << out_transform_size;
  }

#pragma omp parallel for
  for (size_t out_coord_idx = 0; out_coord_idx < out_transform_size;
       ++out_coord_idx) {
    const Coordinate& out_coord = out_coords[out_coord_idx];

    size_t batch_index = out_coord[batch_axis_result];
    size_t output_channel = out_coord[output_channel_axis_result];

    size_t n_spatial_dimensions = arg0_shape.size() - 2;
    size_t n_input_channels = arg0_shape[input_channel_axis_data];

    Coordinate input_batch_transform_start(2 + n_spatial_dimensions);
    Coordinate input_batch_transform_end(2 + n_spatial_dimensions);
    Strides input_batch_transform_movement_strides(2 + n_spatial_dimensions, 1);
    CoordinateDiff input_batch_transform_padding_below(2 + n_spatial_dimensions,
                                                       0);
    CoordinateDiff input_batch_transform_padding_above(2 + n_spatial_dimensions,
                                                       0);
    Strides input_batch_transform_dilation_strides(2 + n_spatial_dimensions, 1);

    input_batch_transform_start[batch_axis_data] = batch_index;
    input_batch_transform_end[batch_axis_data] = batch_index + 1;
    input_batch_transform_start[input_channel_axis_data] = 0;
    input_batch_transform_end[input_channel_axis_data] = n_input_channels;

    for (size_t i = 2; i < n_spatial_dimensions + 2; i++) {
      size_t window_dilation_stride = window_dilation_strides[i - 2];
      size_t window_movement_stride = window_movement_strides[i - 2];
      std::ptrdiff_t below_pad = padding_below[i - 2];
      std::ptrdiff_t above_pad = padding_above[i - 2];
      size_t data_dilation_stride = data_dilation_strides[i - 2];

      input_batch_transform_start[i] = window_movement_stride * out_coord[i];
      input_batch_transform_end[i] =
          input_batch_transform_start[i] +
          (arg1_shape[i] - 1) * window_dilation_stride + 1;
      input_batch_transform_movement_strides[i] = window_dilation_stride;
      input_batch_transform_padding_below[i] = below_pad;
      input_batch_transform_padding_above[i] = above_pad;
      input_batch_transform_dilation_strides[i] = data_dilation_stride;
    }

    AxisVector input_batch_transform_axis_order(2 + n_spatial_dimensions);
    for (size_t i = 0; i < input_batch_transform_axis_order.size(); i++) {
      input_batch_transform_axis_order[i] = i;
    }

    CoordinateTransform input_batch_transform(
        arg0_shape, input_batch_transform_start, input_batch_transform_end,
        input_batch_transform_movement_strides,
        input_batch_transform_axis_order, input_batch_transform_padding_below,
        input_batch_transform_padding_above,
        input_batch_transform_dilation_strides);

    Shape filter_transform_start(2 + n_spatial_dimensions);
    Shape filter_transform_end(2 + n_spatial_dimensions);

    filter_transform_start[output_channel_axis_filters] = output_channel;
    filter_transform_end[output_channel_axis_filters] = output_channel + 1;
    filter_transform_start[input_channel_axis_filters] = 0;
    filter_transform_end[input_channel_axis_filters] = n_input_channels;

    for (size_t i = 2; i < n_spatial_dimensions + 2; i++) {
      filter_transform_start[i] = 0;
      filter_transform_end[i] = arg1_shape[i];
    }

    CoordinateTransform filter_transform(arg1_shape, filter_transform_start,
                                         filter_transform_end);

    CoordinateTransform::Iterator input_it = input_batch_transform.begin();
    CoordinateTransform::Iterator filter_it = filter_transform.begin();
    CoordinateTransform::Iterator input_end = input_batch_transform.end();
    CoordinateTransform::Iterator filter_end = filter_transform.end();

    auto sum = HEPlaintext();
    bool first_add = true;

    while (input_it != input_end && filter_it != filter_end) {
      const Coordinate& input_batch_coord = *input_it;
      Coordinate filter_coord = *filter_it;

      if (rotate_filter) {
        Shape target_shape = filter_transform.get_target_shape();
        for (size_t i = 2; i < filter_coord.size(); i++) {
          filter_coord[i] = target_shape[i] - filter_coord[i] - 1;
        }
      }

      if (input_batch_transform.has_source_coordinate(input_batch_coord)) {
        auto mult_arg0 = arg0[input_batch_transform.index(input_batch_coord)];
        auto mult_arg1 = arg1[filter_transform.index(filter_coord)];
        auto prod = HEPlaintext();

        scalar_multiply_seal(mult_arg0, mult_arg1, prod, element_type,
                             he_seal_backend);
        if (first_add) {
          sum = prod;
          first_add = false;
        } else {
          scalar_add_seal(prod, sum, sum);
        }
      }
      ++input_it;
      ++filter_it;
    }
    if (first_add) {
      out[out_coord_idx] = HEPlaintext(0.f);
    } else {
      // Write the sum back.
      out[out_coord_idx] = sum;
    }
  }
}

}  // namespace he
}  // namespace ngraph
