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

#include "he_backend.hpp"
#include "he_ciphertext.hpp"
#include "he_plaintext.hpp"
#include "kernel/broadcast.hpp"
#include "ngraph/coordinate_transform.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::broadcast(const vector<shared_ptr<he::HECiphertext>>& arg,
                                    vector<shared_ptr<he::HECiphertext>>& out,
                                    const Shape& in_shape,
                                    const Shape& out_shape,
                                    const AxisSet& broadcast_axes)
{
    broadcast<he::HECiphertext, he::HECiphertext>(arg, out, in_shape, out_shape, broadcast_axes);
}

void runtime::he::kernel::broadcast(const vector<shared_ptr<he::HEPlaintext>>& arg,
                                    vector<shared_ptr<he::HEPlaintext>>& out,
                                    const Shape& in_shape,
                                    const Shape& out_shape,
                                    const AxisSet& broadcast_axes)
{
    broadcast<he::HEPlaintext, he::HEPlaintext>(arg, out, in_shape, out_shape, broadcast_axes);
}

void runtime::he::kernel::broadcast(const vector<shared_ptr<he::HEPlaintext>>& arg,
                                    vector<shared_ptr<he::HECiphertext>>& out,
                                    const Shape& in_shape,
                                    const Shape& out_shape,
                                    const AxisSet& broadcast_axes,
                                    shared_ptr<HEBackend> he_backend)
{
    CoordinateTransform input_transform(in_shape);
    CoordinateTransform output_transform(out_shape);
    for (const Coordinate& output_coord : output_transform)
    {
        Coordinate input_coord = project(output_coord, broadcast_axes);

        shared_ptr<he::HECiphertext> c = make_shared<he::HECiphertext>();
        he_backend->encrypt(c, arg[input_transform.index(input_coord)]);
        out[output_transform.index(output_coord)] = c;
    }
}
