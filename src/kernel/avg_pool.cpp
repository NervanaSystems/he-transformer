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
#include "kernel/avg_pool.hpp"
#include "kernel/multiply.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::avg_pool(const vector<shared_ptr<runtime::he::HECiphertext>>& arg,
                                   vector<shared_ptr<runtime::he::HECiphertext>>& out,
                                   const Shape& arg_shape,
                                   const Shape& out_shape,
                                   const Shape& window_shape,
                                   const Strides& window_movement_strides,
                                   const Shape& padding_below,
                                   const Shape& padding_above,
                                   bool include_padding_in_avg_computation,
                                   const element::Type& type,
                                   const shared_ptr<runtime::he::HEBackend> he_backend)
{
    auto he_seal_backend = dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(he_backend);
    auto he_heaan_backend = dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(he_backend);
    if (!he_seal_backend && !he_heaan_backend)
    {
        throw ngraph_error("Convolution he_backend neither seal nor hean.");
    }

    // At the outermost level we will walk over every output coordinate O.
    CoordinateTransform output_transform(out_shape);

    size_t out_transform_size = 0;
    for (Coordinate out_coord : output_transform)
    {
        out_transform_size++;
    }

#pragma omp parallel for
    for (size_t out_coord_idx = 0; out_coord_idx < out_transform_size; ++out_coord_idx)
    {
        // TODO: move to coordinate transform
        auto out_coord_it = output_transform.begin();
        for (size_t i = 0; i < out_coord_idx; ++i)
        {
            ++out_coord_it;
        }

        const Coordinate out_coord = *out_coord_it;

        //for (const Coordinate& out_coord : output_transform)
        //{
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
        // We iterate this over the *padded* data, so below we will need to check for coordinates that fall in the padding area.

        size_t n_spatial_dimensions = arg_shape.size() - 2;

        Coordinate input_batch_transform_start(2 + n_spatial_dimensions);
        Coordinate input_batch_transform_end(2 + n_spatial_dimensions);
        Strides input_batch_transform_source_strides(2 + n_spatial_dimensions, 1);
        AxisVector input_batch_transform_source_axis_order(2 + n_spatial_dimensions);
        CoordinateDiff input_batch_transform_padding_below(2 + n_spatial_dimensions);
        CoordinateDiff input_batch_transform_padding_above(2 + n_spatial_dimensions);

        input_batch_transform_start[0] = batch_index;
        input_batch_transform_end[0] = batch_index + 1;
        input_batch_transform_start[1] = channel;
        input_batch_transform_end[1] = channel + 1;
        input_batch_transform_padding_below[0] = 0;
        input_batch_transform_padding_below[1] = 0;
        input_batch_transform_padding_above[0] = 0;
        input_batch_transform_padding_above[1] = 0;

        for (size_t i = 2; i < n_spatial_dimensions + 2; i++)
        {
            size_t window_shape_this_dim = window_shape[i - 2];
            size_t movement_stride = window_movement_strides[i - 2];

            input_batch_transform_start[i] = movement_stride * out_coord[i];
            input_batch_transform_end[i] = input_batch_transform_start[i] + window_shape_this_dim;
            input_batch_transform_padding_below[i] = padding_below[i - 2];
            input_batch_transform_padding_above[i] = padding_above[i - 2];
        }

        for (size_t i = 0; i < arg_shape.size(); i++)
        {
            input_batch_transform_source_axis_order[i] = i;
        }

        CoordinateTransform input_batch_transform(arg_shape,
                                                  input_batch_transform_start,
                                                  input_batch_transform_end,
                                                  input_batch_transform_source_strides,
                                                  input_batch_transform_source_axis_order,
                                                  input_batch_transform_padding_below,
                                                  input_batch_transform_padding_above);

        // As we go, we compute the sum value:
        //
        //   output[O] := output[O] + arg[I]
        //
        // and the number of elements:
        //
        //   n_elements := n_elements + 1

        // T result = 0;
        shared_ptr<runtime::he::HECiphertext> result;
        if (he_seal_backend)
        {
            result = he_seal_backend->create_valued_ciphertext(0., type);
        }
        else if (he_heaan_backend)
        {
            result = he_heaan_backend->create_valued_ciphertext(0., type);
        }
        size_t n_elements = 0;

        for (const Coordinate& input_batch_coord : input_batch_transform)
        {
            bool in_bounds = input_batch_transform.has_source_coordinate(input_batch_coord);

            if (in_bounds || include_padding_in_avg_computation)
            {
                //T v =
                //	in_bounds ? arg[input_batch_transform.index(input_batch_coord)] : 0;
                shared_ptr<runtime::he::HECiphertext> v;

                if (he_seal_backend)
                {
                    v = input_batch_transform.has_source_coordinate(input_batch_coord)
                            ? arg[input_batch_transform.index(input_batch_coord)]
                            : he_seal_backend->create_valued_ciphertext(0., type);
                }
                else if (he_heaan_backend)
                {
                    v = input_batch_transform.has_source_coordinate(input_batch_coord)
                            ? arg[input_batch_transform.index(input_batch_coord)]
                            : he_heaan_backend->create_valued_ciphertext(0., type);
                }

                // result += v;
                runtime::he::kernel::scalar_add(result, v, result, type, he_backend);

                n_elements++;
            }
        }
        shared_ptr<runtime::he::HEPlaintext> inv_n_elements;
        if (he_seal_backend)
        {
            inv_n_elements = he_seal_backend->create_valued_plaintext(1. / n_elements, type);
        }
        else if (he_heaan_backend)
        {
            inv_n_elements = he_heaan_backend->create_valued_plaintext(1. / n_elements, type);
        }

        runtime::he::kernel::scalar_multiply(result, inv_n_elements, result, type, he_backend);

        out[output_transform.index(out_coord)] = result;
    }
}

void runtime::he::kernel::avg_pool(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg,
                                   vector<shared_ptr<runtime::he::HEPlaintext>>& out,
                                   const Shape& arg_shape,
                                   const Shape& out_shape,
                                   const Shape& window_shape,
                                   const Strides& window_movement_strides,
                                   const Shape& padding_below,
                                   const Shape& padding_above,
                                   bool include_padding_in_avg_computation,
                                   const element::Type& type,
                                   const shared_ptr<runtime::he::HEBackend> he_backend)
{
    auto he_seal_backend = dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(he_backend);
    auto he_heaan_backend = dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(he_backend);
    if (!he_seal_backend && !he_heaan_backend)
    {
        throw ngraph_error("Convolution he_backend neither seal nor hean.");
    }

    // At the outermost level we will walk over every output coordinate O.
    CoordinateTransform output_transform(out_shape);

    size_t out_transform_size = 0;
    for (Coordinate out_coord : output_transform)
    {
        out_transform_size++;
    }

#pragma omp parallel for
    for (size_t out_coord_idx = 0; out_coord_idx < out_transform_size; ++out_coord_idx)
    {
        // TODO: move to coordinate transform
        auto out_coord_it = output_transform.begin();
        for (size_t i = 0; i < out_coord_idx; ++i)
        {
            ++out_coord_it;
        }

        const Coordinate out_coord = *out_coord_it;

        //for (const Coordinate& out_coord : output_transform)
        //{
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
        // We iterate this over the *padded* data, so below we will need to check for coordinates that fall in the padding area.

        size_t n_spatial_dimensions = arg_shape.size() - 2;

        Coordinate input_batch_transform_start(2 + n_spatial_dimensions);
        Coordinate input_batch_transform_end(2 + n_spatial_dimensions);
        Strides input_batch_transform_source_strides(2 + n_spatial_dimensions, 1);
        AxisVector input_batch_transform_source_axis_order(2 + n_spatial_dimensions);
        CoordinateDiff input_batch_transform_padding_below(2 + n_spatial_dimensions);
        CoordinateDiff input_batch_transform_padding_above(2 + n_spatial_dimensions);

        input_batch_transform_start[0] = batch_index;
        input_batch_transform_end[0] = batch_index + 1;
        input_batch_transform_start[1] = channel;
        input_batch_transform_end[1] = channel + 1;
        input_batch_transform_padding_below[0] = 0;
        input_batch_transform_padding_below[1] = 0;
        input_batch_transform_padding_above[0] = 0;
        input_batch_transform_padding_above[1] = 0;

        for (size_t i = 2; i < n_spatial_dimensions + 2; i++)
        {
            size_t window_shape_this_dim = window_shape[i - 2];
            size_t movement_stride = window_movement_strides[i - 2];

            input_batch_transform_start[i] = movement_stride * out_coord[i];
            input_batch_transform_end[i] = input_batch_transform_start[i] + window_shape_this_dim;
            input_batch_transform_padding_below[i] = padding_below[i - 2];
            input_batch_transform_padding_above[i] = padding_above[i - 2];
        }

        for (size_t i = 0; i < arg_shape.size(); i++)
        {
            input_batch_transform_source_axis_order[i] = i;
        }

        CoordinateTransform input_batch_transform(arg_shape,
                                                  input_batch_transform_start,
                                                  input_batch_transform_end,
                                                  input_batch_transform_source_strides,
                                                  input_batch_transform_source_axis_order,
                                                  input_batch_transform_padding_below,
                                                  input_batch_transform_padding_above);

        // As we go, we compute the sum value:
        //
        //   output[O] := output[O] + arg[I]
        //
        // and the number of elements:
        //
        //   n_elements := n_elements + 1

        // T result = 0;
        shared_ptr<runtime::he::HEPlaintext> result;
        if (he_seal_backend)
        {
            result = he_seal_backend->create_valued_plaintext(0., type);
        }
        else if (he_heaan_backend)
        {
            result = he_heaan_backend->create_valued_plaintext(0., type);
        }
        size_t n_elements = 0;

        for (const Coordinate& input_batch_coord : input_batch_transform)
        {
            bool in_bounds = input_batch_transform.has_source_coordinate(input_batch_coord);

            if (in_bounds || include_padding_in_avg_computation)
            {
                //T v =
                //	in_bounds ? arg[input_batch_transform.index(input_batch_coord)] : 0;
                shared_ptr<runtime::he::HEPlaintext> v;

                if (he_seal_backend)
                {
                    v = input_batch_transform.has_source_coordinate(input_batch_coord)
                            ? arg[input_batch_transform.index(input_batch_coord)]
                            : he_seal_backend->create_valued_plaintext(0., type);
                }
                else if (he_heaan_backend)
                {
                    v = input_batch_transform.has_source_coordinate(input_batch_coord)
                            ? arg[input_batch_transform.index(input_batch_coord)]
                            : he_heaan_backend->create_valued_plaintext(0., type);
                }

                // result += v;
                runtime::he::kernel::scalar_add(result, v, result, type, he_backend);

                n_elements++;
            }
        }
        // TODO: use divide op? Or don't divide?
        shared_ptr<runtime::he::HEPlaintext> inv_n_elements;
        if (he_seal_backend)
        {
            inv_n_elements = he_seal_backend->create_valued_plaintext(1. / n_elements, type);
        }
        else if (he_heaan_backend)
        {
            inv_n_elements = he_heaan_backend->create_valued_plaintext(1. / n_elements, type);
        }

        runtime::he::kernel::scalar_multiply(result, inv_n_elements, result, type, he_backend);

        out[output_transform.index(out_coord)] = result;
    }
}
