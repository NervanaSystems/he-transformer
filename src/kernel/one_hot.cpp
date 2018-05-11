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
#include "he_cipher_tensor_view.hpp"
#include "kernel/one_hot.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/seal.h"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::one_hot(const vector<shared_ptr<seal::Ciphertext>>& arg,
                                  vector<shared_ptr<seal::Ciphertext>>& out,
                                  const Shape& in_shape,
                                  const Shape& out_shape,
                                  size_t one_hot_axis,
                                  const element::Type& type,
                                  shared_ptr<HEBackend>& he_backend)
{
    // Step 1: Zero out the output.
    CoordinateTransform output_transform(out_shape);
    for (const Coordinate& output_coord : output_transform)
    {
        shared_ptr<HECipherTensorView> zero_tv =
            static_pointer_cast<HECipherTensorView>(he_backend->create_zero_tensor(type, Shape{1}));

        out[output_transform.index(output_coord)] = zero_tv->get_element(0);
    }

    // Step 2: Write ones at needed positions, throwing exceptions when invalid conditions
    // are encountered.
    CoordinateTransform input_transform(in_shape);

    for (const Coordinate& input_coord : input_transform)
    {
        shared_ptr<seal::Ciphertext> val = arg[input_transform.index(input_coord)];

        size_t one_hot_pos = out_shape[one_hot_axis] + 1;
        for (size_t i = 0; i < out_shape[one_hot_axis]; ++i)
        {
            shared_ptr<HECipherTensorView> const_tv = static_pointer_cast<HECipherTensorView>(
                he_backend->create_constant_tensor(type, Shape{1}, i));
            seal::Plaintext dec_val;
            seal::Plaintext dec_i;
            // TODO: We are not allowed to decrypt! Pass in one-hot encoded inputs
            he_backend->decrypt(dec_val, *val);
            // TODO: We are not allowed to decrypt! Pass in one-hot encoded inputs
            he_backend->decrypt(dec_i, *(const_tv->get_element(0)));

            if (dec_val == dec_i)
            {
                one_hot_pos = i;
                break;
            }
        }
        if (one_hot_pos == out_shape[one_hot_axis] + 1)
        {
            throw(std::range_error(
                "One-hot: non-integral value in input or value is out of category range"));
        }

        Coordinate one_hot_coord = inject(input_coord, one_hot_axis, one_hot_pos);
        shared_ptr<HECipherTensorView> one_tv =
            static_pointer_cast<HECipherTensorView>(he_backend->create_ones_tensor(type, Shape{1}));

        out[output_transform.index(one_hot_coord)] = one_tv->get_element(0);
    }
}
