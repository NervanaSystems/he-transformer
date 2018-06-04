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

#include <cmath>
#include <utility>

#include "he_backend.hpp"
#include "he_cipher_tensor_view.hpp"
#include "he_heaan_backend.hpp"
#include "he_plain_tensor_view.hpp"
#include "he_seal_backend.hpp"
#include "kernel/add.hpp"
#include "kernel/dot.hpp"
#include "kernel/multiply.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::dot(const vector<shared_ptr<runtime::he::HECiphertext>>& arg0,
                              const vector<shared_ptr<runtime::he::HECiphertext>>& arg1,
                              vector<shared_ptr<runtime::he::HECiphertext>>& out,
                              const Shape& arg0_shape,
                              const Shape& arg1_shape,
                              const Shape& out_shape,
                              size_t reduction_axes_count,
                              const element::Type& type,
                              const shared_ptr<runtime::he::HEBackend>& he_backend)
{
    dot_template(
        arg0, arg1, out, arg0_shape, arg1_shape, out_shape, reduction_axes_count, type, he_backend);
}

void runtime::he::kernel::dot(const vector<shared_ptr<runtime::he::HECiphertext>>& arg0,
                              const vector<shared_ptr<runtime::he::HEPlaintext>>& arg1,
                              vector<shared_ptr<runtime::he::HECiphertext>>& out,
                              const Shape& arg0_shape,
                              const Shape& arg1_shape,
                              const Shape& out_shape,
                              size_t reduction_axes_count,
                              const element::Type& type,
                              const shared_ptr<runtime::he::HEBackend>& he_backend)
{
    dot_template(
        arg0, arg1, out, arg0_shape, arg1_shape, out_shape, reduction_axes_count, type, he_backend);
}

void runtime::he::kernel::dot(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg0,
                              const vector<shared_ptr<runtime::he::HECiphertext>>& arg1,
                              vector<shared_ptr<runtime::he::HECiphertext>>& out,
                              const Shape& arg0_shape,
                              const Shape& arg1_shape,
                              const Shape& out_shape,
                              size_t reduction_axes_count,
                              const element::Type& type,
                              const shared_ptr<runtime::he::HEBackend>& he_backend)
{
    dot_template(
        arg0, arg1, out, arg0_shape, arg1_shape, out_shape, reduction_axes_count, type, he_backend);
}

// TODO: merge with template dot
void runtime::he::kernel::dot(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg0,
                              const vector<shared_ptr<runtime::he::HEPlaintext>>& arg1,
                              vector<shared_ptr<runtime::he::HEPlaintext>>& out,
                              const Shape& arg0_shape,
                              const Shape& arg1_shape,
                              const Shape& out_shape,
                              size_t reduction_axes_count,
                              const element::Type& type,
                              const shared_ptr<runtime::he::HEBackend>& he_backend)
{
    auto he_seal_backend = dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(he_backend);
    auto he_heaan_backend = dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(he_backend);
    if (!he_seal_backend && !he_heaan_backend)
    {
        throw ngraph_error("Dot he_backend neither seal nor hean;");
    }
    // Get the sizes of the dot axes. It's easiest to pull them from arg1 because they're
    // right up front.
    Shape dot_axis_sizes(reduction_axes_count);
    copy(arg1_shape.begin(), arg1_shape.begin() + reduction_axes_count, dot_axis_sizes.begin());

    CoordinateTransform arg0_transform(arg0_shape);
    CoordinateTransform arg1_transform(arg1_shape);
    CoordinateTransform output_transform(out_shape);

    // Create coordinate transforms for arg0 and arg1 that throw away the dotted axes.
    size_t arg0_projected_rank = arg0_shape.size() - reduction_axes_count;
    size_t arg1_projected_rank = arg1_shape.size() - reduction_axes_count;

    Shape arg0_projected_shape(arg0_projected_rank);
    copy(
        arg0_shape.begin(), arg0_shape.begin() + arg0_projected_rank, arg0_projected_shape.begin());

    Shape arg1_projected_shape(arg1_projected_rank);
    copy(arg1_shape.begin() + reduction_axes_count, arg1_shape.end(), arg1_projected_shape.begin());

    CoordinateTransform arg0_projected_transform(arg0_projected_shape);
    CoordinateTransform arg1_projected_transform(arg1_projected_shape);

    // Create a coordinate transform that allows us to iterate over all possible values
    // for the dotted axes.
    CoordinateTransform dot_axes_transform(dot_axis_sizes);

    // Get arg0_projected_size and arg1_projected_size for parallelization
    size_t arg0_projected_size = 0;
    for (const Coordinate& arg0_projected_coord : arg0_projected_transform)
    {
        arg0_projected_size++;
    }
    size_t arg1_projected_size = 0;
    for (const Coordinate& arg1_projected_coord : arg1_projected_transform)
    {
        arg1_projected_size++;
    }
    size_t global_projected_size = arg0_projected_size * arg1_projected_size;

#pragma omp parallel for
    for (size_t global_projected_idx = 0; global_projected_idx < global_projected_size;
         ++global_projected_idx)
    {
        // Compute outer and inner index
        size_t arg0_projected_idx = global_projected_idx / arg1_projected_size;
        size_t arg1_projected_idx = global_projected_idx % arg1_projected_size;

        // TODO: move to coordinate transform, or precompute this and store in a
        //       matrix
        auto arg0_projected_it = arg0_projected_transform.begin();
        for (size_t i = 0; i < arg0_projected_idx; ++i)
        {
            ++arg0_projected_it;
        }
        const Coordinate& arg0_projected_coord = *arg0_projected_it;
        auto arg1_projected_it = arg1_projected_transform.begin();
        for (size_t i = 0; i < arg1_projected_idx; ++i)
        {
            ++arg1_projected_it;
        }
        const Coordinate& arg1_projected_coord = *arg1_projected_it;

        // The output coordinate is just the concatenation of the projected coordinates.
        Coordinate out_coord(arg0_projected_coord.size() + arg1_projected_coord.size());

        auto out_coord_it =
            copy(arg0_projected_coord.begin(), arg0_projected_coord.end(), out_coord.begin());
        copy(arg1_projected_coord.begin(), arg1_projected_coord.end(), out_coord_it);

        // Zero out to start the sum
        shared_ptr<runtime::he::HEPlaintext> sum;
        if (he_seal_backend)
        {
            sum = he_seal_backend->create_valued_plaintext(0., type);
        }
        else if (he_heaan_backend)
        {
            sum = he_heaan_backend->create_valued_plaintext(0., type);
        }

        size_t out_index = output_transform.index(out_coord);

        // Walk along the dotted axes.
        Coordinate arg0_coord(arg0_shape.size());
        Coordinate arg1_coord(arg1_shape.size());
        auto arg0_it =
            copy(arg0_projected_coord.begin(), arg0_projected_coord.end(), arg0_coord.begin());

        for (const Coordinate& dot_axis_positions : dot_axes_transform)
        {
            // In order to find the points to multiply together, we need to inject our current
            // positions along the dotted axes back into the projected arg0 and arg1 coordinates.
            copy(dot_axis_positions.begin(), dot_axis_positions.end(), arg0_it);

            auto arg1_it =
                copy(dot_axis_positions.begin(), dot_axis_positions.end(), arg1_coord.begin());
            copy(arg1_projected_coord.begin(), arg1_projected_coord.end(), arg1_it);

            // Multiply and add to the sum.
            auto arg0_text = arg0[arg0_transform.index(arg0_coord)];
            auto arg1_text = arg1[arg1_transform.index(arg1_coord)];

            shared_ptr<runtime::he::HEPlaintext> prod;

            if (he_seal_backend)
            {
                prod = he_seal_backend->create_empty_plaintext();
            }
            else if (he_heaan_backend)
            {
                prod = he_heaan_backend->create_empty_plaintext();
            }

            runtime::he::kernel::scalar_multiply(arg0_text, arg1_text, prod, type, he_backend);
            runtime::he::kernel::scalar_add(sum, prod, sum, type, he_backend);
        }

        // Write the sum back.
        out[out_index] = sum;
    }
}
