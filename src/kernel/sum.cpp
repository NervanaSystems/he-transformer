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

#include "he_backend.hpp"
#include "kernel/sum.hpp"
#include "seal/seal.h"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::sum(const vector<shared_ptr<seal::Ciphertext>>& arg,
                              vector<shared_ptr<seal::Ciphertext>>& out,
                              const AxisSet& reduction_axes)
{
    CoordinateTransform output_transform(out_shape);

    shared_ptr<HECipherTensorView> zero_tv =
        static_pointer_cast<HECipherTensorView>(he_backend->create_zero_tensor(type, Shape{1}));
    shared_ptr<seal::Ciphertext> zero = zero_tv->get_element(0);

    for (const Coordinate& output_coord : output_transform)
    {
        out[output_transform.index(output_coord)] = zero;
    }

    CoordinateTransform input_transform(in_shape);

    for (const Coordinate& input_coord : input_transform)
    {
        Coordinate output_coord = project(input_coord, reduction_axes);

        out[output_transform.index(output_coord)] += arg[input_transform.index(input_coord)];
    }
}
