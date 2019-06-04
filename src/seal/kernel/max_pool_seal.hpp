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

#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "ngraph/coordinate_transform.hpp"
#include "seal/client_util.hpp"

namespace ngraph {
namespace he {
// Returns list where L[i] is the list of input indices to maximize over.
std::vector<std::vector<size_t>> max_pool_seal(
    const Shape& arg_shape, const Shape& out_shape, const Shape& window_shape,
    const Strides& window_movement_strides, const Shape& padding_below,
    const Shape& padding_above) {
  // At the outermost level we will walk over every output coordinate O.
  CoordinateTransform output_transform(out_shape);

  size_t out_size = 0;
  for (const Coordinate& out_coord : output_transform) {
    out_size++;
  }

  std::vector<std::vector<size_t>> maximize_list(shape_size(out_shape));

  for (const Coordinate& out_coord : output_transform) {
    // Our output coordinate O will have the form:
    //
    //   (N,chan,i_1,...,i_n)

    size_t batch_index = out_coord[0];
    size_t channel = out_coord[1];

    // For the input data we need to iterate the coordinate:
    //
    //   I:
    //
    // over the range (noninclusive on the right):
    //
    //   (N,chan,s_1*i_1,s_2*i_2,...,s_n*i_n) ->
    //
    //     (N+1,chan+1,s_1*i_1 + window_shape_1,...,s_n*i_n +
    //     window_shape_n)
    //
    // with unit stride.
    //
    // We iterate this over the *padded* data, so below we will need to
    // check for coordinates that fall in the padding area.

    size_t n_spatial_dimensions = arg_shape.size() - 2;

    Coordinate input_batch_transform_start(2 + n_spatial_dimensions);
    Coordinate input_batch_transform_end(2 + n_spatial_dimensions);
    Strides input_batch_transform_source_strides(2 + n_spatial_dimensions, 1);
    AxisVector input_batch_transform_source_axis_order(2 +
                                                       n_spatial_dimensions);
    CoordinateDiff input_batch_transform_padding_below(2 +
                                                       n_spatial_dimensions);
    CoordinateDiff input_batch_transform_padding_above(2 +
                                                       n_spatial_dimensions);

    input_batch_transform_start[0] = batch_index;
    input_batch_transform_end[0] = batch_index + 1;
    input_batch_transform_start[1] = channel;
    input_batch_transform_end[1] = channel + 1;
    input_batch_transform_padding_below[0] = 0;
    input_batch_transform_padding_below[1] = 0;
    input_batch_transform_padding_above[0] = 0;
    input_batch_transform_padding_above[1] = 0;

    for (size_t i = 2; i < n_spatial_dimensions + 2; i++) {
      size_t window_shape_this_dim = window_shape[i - 2];
      size_t movement_stride = window_movement_strides[i - 2];

      input_batch_transform_start[i] = movement_stride * out_coord[i];
      input_batch_transform_end[i] =
          input_batch_transform_start[i] + window_shape_this_dim;
      input_batch_transform_padding_below[i] = padding_below[i - 2];
      input_batch_transform_padding_above[i] = padding_above[i - 2];
    }

    for (size_t i = 0; i < arg_shape.size(); i++) {
      input_batch_transform_source_axis_order[i] = i;
    }

    CoordinateTransform input_batch_transform(
        arg_shape, input_batch_transform_start, input_batch_transform_end,
        input_batch_transform_source_strides,
        input_batch_transform_source_axis_order,
        input_batch_transform_padding_below,
        input_batch_transform_padding_above);

    // As we go, we compute the maximum value:
    //
    //   output[O] = max(output[O],arg[I])

    size_t out_index = output_transform.index(out_coord);
    for (const Coordinate& input_batch_coord : input_batch_transform) {
      if (input_batch_transform.has_source_coordinate(input_batch_coord)) {
        int new_index = input_batch_transform.index(input_batch_coord);
        maximize_list[out_index].emplace_back(new_index);
      }
    }
  }
  return maximize_list;
}

void max_pool_seal(const std::vector<HEPlaintext>& arg,
                   std::vector<HEPlaintext>& out, const Shape& arg_shape,
                   const Shape& out_shape, const Shape& window_shape,
                   const Strides& window_movement_strides,
                   const Shape& padding_below, const Shape& padding_above) {
  NGRAPH_INFO << "Maxpool seal";
  NGRAPH_INFO << "arg_shape " << join(arg_shape, "x");
  NGRAPH_INFO << "out_shape " << join(out_shape, "x");
  // At the outermost level we will walk over every output coordinate O.
  CoordinateTransform output_transform(out_shape);

  size_t out_coord_idx = 0;

  for (const Coordinate& out_coord : output_transform) {
    out_coord_idx++;
    // Our output coordinate O will have the form:
    //
    //   (N,chan,i_1,...,i_n)

    size_t batch_index = out_coord[0];
    size_t channel = out_coord[1];

    // For the input data we need to iterate the coordinate:
    //
    //   I:
    //
    // over the range (noninclusive on the right):
    //
    //   (N,chan,s_1*i_1,s_2*i_2,...,s_n*i_n) ->
    //
    //     (N+1,chan+1,s_1*i_1 + window_shape_1,...,s_n*i_n + window_shape_n)
    //
    // with unit stride.
    //
    // We iterate this over the *padded* data, so below we will need to check
    // for coordinates that fall in the padding area.

    size_t n_spatial_dimensions = arg_shape.size() - 2;

    Coordinate input_batch_transform_start(2 + n_spatial_dimensions);
    Coordinate input_batch_transform_end(2 + n_spatial_dimensions);
    Strides input_batch_transform_source_strides(2 + n_spatial_dimensions, 1);
    AxisVector input_batch_transform_source_axis_order(2 +
                                                       n_spatial_dimensions);
    CoordinateDiff input_batch_transform_padding_below(2 +
                                                       n_spatial_dimensions);
    CoordinateDiff input_batch_transform_padding_above(2 +
                                                       n_spatial_dimensions);

    input_batch_transform_start[0] = batch_index;
    input_batch_transform_end[0] = batch_index + 1;
    input_batch_transform_start[1] = channel;
    input_batch_transform_end[1] = channel + 1;
    input_batch_transform_padding_below[0] = 0;
    input_batch_transform_padding_below[1] = 0;
    input_batch_transform_padding_above[0] = 0;
    input_batch_transform_padding_above[1] = 0;

    for (size_t i = 2; i < n_spatial_dimensions + 2; i++) {
      size_t window_shape_this_dim = window_shape[i - 2];
      size_t movement_stride = window_movement_strides[i - 2];

      input_batch_transform_start[i] = movement_stride * out_coord[i];
      input_batch_transform_end[i] =
          input_batch_transform_start[i] + window_shape_this_dim;
      input_batch_transform_padding_below[i] = padding_below[i - 2];
      input_batch_transform_padding_above[i] = padding_above[i - 2];
    }

    for (size_t i = 0; i < arg_shape.size(); i++) {
      input_batch_transform_source_axis_order[i] = i;
    }

    CoordinateTransform input_batch_transform(
        arg_shape, input_batch_transform_start, input_batch_transform_end,
        input_batch_transform_source_strides,
        input_batch_transform_source_axis_order,
        input_batch_transform_padding_below,
        input_batch_transform_padding_above);

    // As we go, we compute the maximum value:
    //
    //   output[O] = max(output[O],arg[I])
    bool first_max = true;
    std::vector<float> max_vals;

    for (const Coordinate& input_batch_coord : input_batch_transform) {
      if (input_batch_transform.has_source_coordinate(input_batch_coord)) {
        auto arg_coord_idx = input_batch_transform.index(input_batch_coord);
        if (first_max) {
          first_max = false;
          max_vals = arg[arg_coord_idx].get_values();
          NGRAPH_INFO << "first max";
        } else {
          NGRAPH_INFO << "setting max values";
          // Get element-wise maximump
          NGRAPH_INFO << "current max";
          for (const auto& elem : max_vals) {
            NGRAPH_INFO << elem;
          }
          const std::vector<float>& arg_values =
              arg[arg_coord_idx].get_values();
          NGRAPH_INFO << "arg values";
          for (const auto& elem : arg_values) {
            NGRAPH_INFO << elem;
          }
          NGRAPH_CHECK(arg_values.size() == max_vals.size(), "arg values size ",
                       arg_values.size(), " doesn't match max_vals.size() ",
                       max_vals.size());
          for (size_t value_idx = 0; value_idx < arg_values.size();
               ++value_idx) {
            max_vals[value_idx] =
                std::max(max_vals[value_idx], arg_values[value_idx]);
          }

          NGRAPH_INFO << "max_vals";
          for (const auto& elem : max_vals) {
            NGRAPH_INFO << elem;
          }
        }
      }
    }
    HEPlaintext result(max_vals);
    out[output_transform.index(out_coord)] = result;
  }
}

}  // namespace he
}  // namespace ngraph
