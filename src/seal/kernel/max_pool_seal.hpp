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
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {
// Returns list where L[i] is the list of input indices to maximize over.
inline std::vector<std::vector<size_t>> max_pool_seal(
    const Shape& arg_shape, const Shape& out_shape, const Shape& window_shape,
    const Strides& window_movement_strides, const Shape& padding_below,
    const Shape& padding_above) {
  // At the outermost level we will walk over every output coordinate O.
  CoordinateTransform output_transform(out_shape);

  size_t out_size = 0;
  for (const Coordinate& out_coord : output_transform) {
    (void)out_coord;  // Avoid unused-variable warning
    out_size++;
  }
  NGRAPH_CHECK(out_size == shape_size(out_shape), "out size ", out_size,
               " != shape_size(out_shape) ", join(out_shape, "x"));

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

inline void max_pool_seal(const std::vector<HEPlaintext>& arg,
                          std::vector<HEPlaintext>& out, const Shape& arg_shape,
                          const Shape& out_shape, const Shape& window_shape,
                          const Strides& window_movement_strides,
                          const Shape& padding_below,
                          const Shape& padding_above) {
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
    std::vector<double> max_vals;

    for (const Coordinate& input_batch_coord : input_batch_transform) {
      if (input_batch_transform.has_source_coordinate(input_batch_coord)) {
        auto arg_coord_idx = input_batch_transform.index(input_batch_coord);

        const std::vector<double>& arg_vals = arg[arg_coord_idx].values();

        if (first_max) {
          first_max = false;
          max_vals = arg_vals;
        } else {
          // Get element-wise maximum
          NGRAPH_CHECK(arg_vals.size() == max_vals.size(), "arg values size ",
                       arg_vals.size(), " doesn't match max_vals.size() ",
                       max_vals.size());
          for (size_t value_idx = 0; value_idx < arg_vals.size(); ++value_idx) {
            max_vals[value_idx] =
                std::max(max_vals[value_idx], arg_vals[value_idx]);
          }
        }
      }
    }
    HEPlaintext result(max_vals);
    out[output_transform.index(out_coord)] = result;
  }
}

inline void max_pool_seal(
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const Shape& arg_shape, const Shape& out_shape, const Shape& window_shape,
    const Strides& window_movement_strides, const Shape& padding_below,
    const Shape& padding_above, const seal::parms_id_type& parms_id,
    double scale, seal::CKKSEncoder& ckks_encoder, seal::Encryptor& encryptor,
    seal::Decryptor& decryptor, bool complex_packing) {
  // At the outermost level we will walk over every output coordinate O.
  CoordinateTransform output_transform(out_shape);

  size_t out_coord_idx = 0;

  for (const Coordinate& out_coord : output_transform) {
    out_coord_idx++;

    size_t batch_index = out_coord[0];
    size_t channel = out_coord[1];
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
    std::vector<double> max_vals;

    for (const Coordinate& input_batch_coord : input_batch_transform) {
      if (input_batch_transform.has_source_coordinate(input_batch_coord)) {
        auto arg_coord_idx = input_batch_transform.index(input_batch_coord);
        HEPlaintext plain;

        ngraph::he::decrypt(plain, *arg[arg_coord_idx], decryptor,
                            ckks_encoder);
        const std::vector<double>& arg_vals = plain.values();
        if (first_max) {
          first_max = false;
          max_vals = arg_vals;
        } else {
          // Get element-wise maximum
          NGRAPH_CHECK(arg_vals.size() == max_vals.size(), "arg values size ",
                       arg_vals.size(), " doesn't match max_vals.size() ",
                       max_vals.size());
          for (size_t value_idx = 0; value_idx < arg_vals.size(); ++value_idx) {
            max_vals[value_idx] =
                std::max(max_vals[value_idx], arg_vals[value_idx]);
          }
        }
      }
    }
    HEPlaintext result(max_vals);
    auto cipher = HESealBackend::create_empty_ciphertext(complex_packing);
    ngraph::he::encrypt(cipher, result, parms_id, ngraph::element::f32, scale,
                        ckks_encoder, encryptor, complex_packing);
    out[output_transform.index(out_coord)] = cipher;
  }
}

inline void max_pool_seal(
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const Shape& arg_shape, const Shape& out_shape, const Shape& window_shape,
    const Strides& window_movement_strides, const Shape& padding_below,
    const Shape& padding_above, const HESealBackend& he_seal_backend) {
  max_pool_seal(
      arg, out, arg_shape, out_shape, window_shape, window_movement_strides,
      padding_below, padding_above,
      he_seal_backend.get_context()->first_parms_id(),
      he_seal_backend.get_scale(), *he_seal_backend.get_ckks_encoder(),
      *he_seal_backend.get_encryptor(), *he_seal_backend.get_decryptor(),
      he_seal_backend.complex_packing());
}

}  // namespace he
}  // namespace ngraph
