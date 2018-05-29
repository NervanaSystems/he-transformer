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

#include "kernel/reshape.hpp"
#include "he_backend.hpp"
#include "he_ciphertext.hpp"
#include "he_plaintext.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::reshape(const vector<shared_ptr<HECiphertext>>& arg,
                                  vector<shared_ptr<he::HECiphertext>>& out,
                                  const Shape& in_shape,
                                  const AxisVector& in_axis_order,
                                  const Shape& out_shape)
{
    // Unfortunately we don't yet have a constructor for CoordinateTransform that lets us pass only source_space_shape
    // and source_axis_order so we have to construct the defaults here.
    Shape in_start_corner(in_shape.size(), 0); // (0,...0)
    Strides in_strides(in_shape.size(), 1);    // (1,...,1)

    CoordinateTransform input_transform(
        in_shape, in_start_corner, in_shape, in_strides, in_axis_order);

    CoordinateTransform output_transform(out_shape);
    CoordinateTransform::Iterator output_it = output_transform.begin();

    for (const Coordinate& input_coord : input_transform)
    {
        const Coordinate& output_coord = *output_it;

        out[output_transform.index(output_coord)] = arg[input_transform.index(input_coord)];

        ++output_it;
    }
}

void runtime::he::kernel::reshape(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg,
                                  vector<shared_ptr<runtime::he::HEPlaintext>>& out,
                                  const Shape& in_shape,
                                  const AxisVector& in_axis_order,
                                  const Shape& out_shape)
{
    // Unfortunately we don't yet have a constructor for CoordinateTransform that lets us pass only source_space_shape
    // and source_axis_order so we have to construct the defaults here.
    Shape in_start_corner(in_shape.size(), 0); // (0,...0)
    Strides in_strides(in_shape.size(), 1);    // (1,...,1)

    CoordinateTransform input_transform(
        in_shape, in_start_corner, in_shape, in_strides, in_axis_order);

    CoordinateTransform output_transform(out_shape);
    CoordinateTransform::Iterator output_it = output_transform.begin();

    for (const Coordinate& input_coord : input_transform)
    {
        const Coordinate& output_coord = *output_it;

        out[output_transform.index(output_coord)] = arg[input_transform.index(input_coord)];

        ++output_it;
    }
}

void runtime::he::kernel::reshape(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg,
                                  vector<shared_ptr<he::HECiphertext>>& out,
                                  const Shape& in_shape,
                                  const AxisVector& in_axis_order,
                                  const Shape& out_shape,
                                  shared_ptr<runtime::he::HEBackend> he_backend)
{
    // Unfortunately we don't yet have a constructor for CoordinateTransform that lets us pass only source_space_shape
    // and source_axis_order so we have to construct the defaults here.
    Shape in_start_corner(in_shape.size(), 0); // (0,...0)
    Strides in_strides(in_shape.size(), 1);    // (1,...,1)

    CoordinateTransform input_transform(
        in_shape, in_start_corner, in_shape, in_strides, in_axis_order);

    CoordinateTransform output_transform(out_shape);
    CoordinateTransform::Iterator output_it = output_transform.begin();

    size_t it_size = 0;
    for (const Coordinate& output_it : output_transform)
    {
        it_size++;
    }

#pragma omp parallel for
    for (size_t i = 0; i < it_size; ++i)
    {
        // TODO: move to coordinate transform
        auto input_it = input_transform.begin();
        auto output_it = output_transform.begin();
        for (size_t j = 0; j < i; ++j)
        {
            ++input_it;
            ++output_it;
        }
        const Coordinate& input_coord = *input_it;
        const Coordinate& output_coord = *output_it;

        shared_ptr<he::HECiphertext> c = make_shared<he::HECiphertext>();
        he_backend->encrypt(c, arg[input_transform.index(input_coord)]);
        out[output_transform.index(output_coord)] = c;
    }
}
