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

#include "he_backend.hpp"
#include "he_cipher_tensor_view.hpp"
#include "he_ciphertext.hpp"
#include "he_heaan_backend.hpp"
#include "he_plaintext.hpp"
#include "he_seal_backend.hpp"
#include "kernel/one_hot.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::one_hot(const vector<shared_ptr<runtime::he::HECiphertext>>& arg,
                                  vector<shared_ptr<runtime::he::HECiphertext>>& out,
                                  const Shape& in_shape,
                                  const Shape& out_shape,
                                  size_t one_hot_axis,
                                  const element::Type& type,
                                  const shared_ptr<runtime::he::HEBackend>& he_backend)
{
    auto he_seal_backend = dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(he_backend);
    auto he_heaan_backend = dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(he_backend);
    if (!he_seal_backend && !he_heaan_backend)
    {
        throw ngraph_error("One-Hot he_backend neither seal nor heaan");
    }
    // Get 0 and 1 cipher text
    shared_ptr<runtime::he::HECiphertext> zero_ciphertext;
    shared_ptr<runtime::he::HECiphertext> one_ciphertext;

    if (he_seal_backend)
    {
        zero_ciphertext = he_seal_backend->create_valued_ciphertext(0, type);
        one_ciphertext = he_seal_backend->create_valued_ciphertext(1, type);
    }
    else if (he_heaan_backend)
    {
        zero_ciphertext = he_heaan_backend->create_valued_ciphertext(0, type);
        one_ciphertext = he_heaan_backend->create_valued_ciphertext(1, type);
    }

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
        shared_ptr<runtime::he::HECiphertext> val = arg[input_transform.index(input_coord)];

        // TODO: We are not allowed to decrypt! Pass in one-hot encoded inputs
        shared_ptr<runtime::he::HEPlaintext> plain_val;
        if (he_heaan_backend)
        {
            plain_val = make_shared<runtime::he::HeaanPlaintextWrapper>();
            he_heaan_backend->decrypt(plain_val, val);
        }
        else if (he_seal_backend)
        {
            plain_val = make_shared<runtime::he::SealPlaintextWrapper>();
            he_seal_backend->decrypt(plain_val, val);
        }
        size_t one_hot_pos;
        const string type_name = type.c_type_string();
        if (type_name == "int64_t")
        {
            int64_t x;
            if (he_seal_backend)
            {
                he_seal_backend->scalar_decode((void*)(&x), plain_val, type);
            }
            else if (he_heaan_backend)
            {
                he_heaan_backend->scalar_decode((void*)(&x), plain_val, type);
            }
            one_hot_pos = static_cast<size_t>(x);
        }
        else if (type_name == "float")
        {
            float x;
            if (he_seal_backend)
            {
                he_seal_backend->scalar_decode((void*)(&x), plain_val, type);
            }
            else if (he_heaan_backend)
            {
                he_heaan_backend->scalar_decode((void*)(&x), plain_val, type);
            }
            if (abs(x - round(x)) > 3e-9)
            {
                NGRAPH_INFO << abs(x - floor(x)) << " diff";
                throw(range_error("One-hot: non-integral value in input"));
            }
            one_hot_pos = static_cast<size_t>(x);
        }
        else
        {
            NGRAPH_INFO << "Unsupported element type in scalar_decode " << type_name;
            throw ngraph_error("Unsupported element type " + type_name);
        }

        if (one_hot_pos >= out_shape[one_hot_axis])
        {
            throw(range_error("One-hot: value is out of category range"));
        }

        Coordinate one_hot_coord = inject(input_coord, one_hot_axis, one_hot_pos);
        out[output_transform.index(one_hot_coord)] = one_ciphertext;
    }
}
