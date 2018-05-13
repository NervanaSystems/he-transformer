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
    // Get 0 and 1 cipher text
    shared_ptr<HECipherTensorView> zero_tv =
        static_pointer_cast<HECipherTensorView>(he_backend->create_zero_tensor(type, Shape{1}));
    shared_ptr<seal::Ciphertext> zero_ciphertext = zero_tv->get_element(0);
    shared_ptr<HECipherTensorView> one_tv =
        static_pointer_cast<HECipherTensorView>(he_backend->create_ones_tensor(type, Shape{1}));
    shared_ptr<seal::Ciphertext> one_ciphertext = one_tv->get_element(0);

    // Step 1: Zero out the output. We can simply copy the shared_ptr pointing to a zero
    // ciphertext to all output locations.
    CoordinateTransform output_transform(out_shape);
    for (const Coordinate& output_coord : output_transform)
    {
        out[output_transform.index(output_coord)] = zero_ciphertext;
    }

    // Step 2: Write ones at needed positions, throwing exceptions when invalid conditions
    // are encountered.
    CoordinateTransform input_transform(in_shape);
    for (const Coordinate& input_coord : input_transform)
    {
        shared_ptr<seal::Ciphertext> val = arg[input_transform.index(input_coord)];

        // TODO: We are not allowed to decrypt! Pass in one-hot encoded inputs
        seal::Plaintext plain_val;
        he_backend->decrypt(plain_val, *val);
        size_t one_hot_pos;

        // TODO: We are not allowed to decrypt and decode
        const string type_name = type.c_type_string();
        if (type_name == "int64_t")
        {
            int64_t x;
            he_backend->decode((void*)(&x), plain_val, type);
            one_hot_pos = static_cast<size_t>(x);
        }
        else if (type_name == "uint64_t")
        {
            uint64_t x;
            he_backend->decode((void*)(&x), plain_val, type);
            one_hot_pos = static_cast<size_t>(x);
        }
        else if (type_name == "float")
        {
            float x;
            he_backend->decode((void*)(&x), plain_val, type);
            if (std::floor(x) < x || std::floor(x) > x)
            {
                throw(std::range_error("One-hot: non-integral value in input"));
            }
            one_hot_pos = static_cast<size_t>(x);
        }
        else
        {
            NGRAPH_INFO << "Unsupported element type in decode " << type_name;
            throw ngraph_error("Unsupported element type " + type_name);
        }

        if (one_hot_pos >= out_shape[one_hot_axis])
        {
            throw(std::range_error("One-hot: value is out of category range"));
        }

        Coordinate one_hot_coord = inject(input_coord, one_hot_axis, one_hot_pos);
        out[output_transform.index(one_hot_coord)] = one_ciphertext;
    }
}
