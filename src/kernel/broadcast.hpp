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

#pragma once

#include "ngraph/coordinate_transform.hpp"
#include "seal/seal.h"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            class HEBackend;

            namespace kernel
            {
                template <typename S, typename T>
                void broadcast(const vector<shared_ptr<S>>& arg,
                               vector<shared_ptr<T>>& out,
                               const Shape& in_shape,
                               const Shape& out_shape,
                               const AxisSet& broadcast_axes)
                {
                    CoordinateTransform input_transform(in_shape);
                    CoordinateTransform output_transform(out_shape);
                    for (const Coordinate& output_coord : output_transform)
                    {
                        Coordinate input_coord = project(output_coord, broadcast_axes);

                        out[output_transform.index(output_coord)] =
                            arg[input_transform.index(input_coord)];
                    }
                };

                void broadcast(const vector<shared_ptr<seal::Ciphertext>>& arg0,
                               vector<shared_ptr<seal::Ciphertext>>& out,
                               const Shape& in_shape,
                               const Shape& out_shape,
                               const AxisSet& broadcast_axes);

                void broadcast(const vector<shared_ptr<seal::Plaintext>>& arg0,
                               vector<shared_ptr<seal::Plaintext>>& out,
                               const Shape& in_shape,
                               const Shape& out_shape,
                               const AxisSet& broadcast_axes);

                void broadcast(const vector<shared_ptr<seal::Plaintext>>& arg0,
                               vector<shared_ptr<seal::Ciphertext>>& out,
                               const Shape& in_shape,
                               const Shape& out_shape,
                               const AxisSet& broadcast_axes,
                               shared_ptr<HEBackend> he_backend);
            }
        }
    }
}
