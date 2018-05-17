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
#include "he_ciphertext.hpp"
#include "he_cipher_tensor_view.hpp"
#include "kernel/add.hpp"
#include "kernel/sum.hpp"
#include "seal/seal.h"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::sum(const vector<shared_ptr<he::HECiphertext>>& arg,
                              vector<shared_ptr<he::HECiphertext>>& out,
                              const Shape& in_shape,
                              const Shape& out_shape,
                              const AxisSet& reduction_axes,
                              const element::Type& type,
                              shared_ptr<HEBackend> he_backend)
{
    CoordinateTransform output_transform(out_shape);

    shared_ptr<HECipherTensorView> zero_tv = static_pointer_cast<HECipherTensorView>(
        he_backend->create_valued_tensor(0., type, out_shape));

    size_t zero_ind = 0;
    vector<vector<he::HECiphertext>> output_summands;
    for (const Coordinate& output_coord : output_transform)
    {
        out[output_transform.index(output_coord)] = zero_tv->get_element(zero_ind);
        output_summands.push_back({});
        ++zero_ind;
    }

    CoordinateTransform input_transform(in_shape);

    for (const Coordinate& input_coord : input_transform)
    {
        Coordinate output_coord = project(input_coord, reduction_axes);
        size_t output_ind = output_transform.index(output_coord);

        shared_ptr<he::HECiphertext> cipher_out = out[output_ind];

        output_summands[output_ind].push_back(*arg[input_transform.index(input_coord)]);

        //ngraph::runtime::he::kernel::add(
        //    cipher_out, arg[input_transform.index(input_coord)], cipher_out, he_backend);
    }

    for (const Coordinate& output_coord : output_transform)
    {
        size_t output_ind = output_transform.index(output_coord);
        if (output_summands[output_ind].size() > 0)
        {
            he_backend->get_evaluator()->add_many(output_summands[output_ind], *out[output_ind]);
        }
    }
}
