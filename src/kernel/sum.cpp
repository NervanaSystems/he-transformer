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
#include "kernel/add.hpp"
#include "seal/seal.h"
#include "he_cipher_tensor_view.hpp"
#include "he_plain_tensor_view.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::sum(const vector<shared_ptr<seal::Ciphertext>>& arg,
                              vector<shared_ptr<seal::Ciphertext>>& out,
                              const Shape& in_shape,
                              const Shape& out_shape,
                              const AxisSet& reduction_axes,
                              const element::Type& type,
                              shared_ptr<HEBackend> he_backend)
{
    CoordinateTransform output_transform(out_shape);

    shared_ptr<HECipherTensorView> zero_tv =
        static_pointer_cast<HECipherTensorView>(he_backend->create_zero_tensor(type, out_shape));

    size_t zero_ind = 0;
    for (const Coordinate& output_coord : output_transform)
    {
        out[output_transform.index(output_coord)] = zero_tv->get_element(zero_ind);
        ++zero_ind;
    }

    CoordinateTransform input_transform(in_shape);

    for (const Coordinate& input_coord : input_transform)
    {
        Coordinate output_coord = project(input_coord, reduction_axes);

        shared_ptr<seal::Ciphertext> cipher_out = out[output_transform.index(output_coord)];

        ngraph::runtime::he::kernel::add(cipher_out, arg[input_transform.index(input_coord)], cipher_out, he_backend);
    }
}
