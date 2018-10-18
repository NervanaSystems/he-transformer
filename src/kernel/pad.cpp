//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <cmath>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/except.hpp"

#include "he_heaan_backend.hpp"
#include "he_seal_backend.hpp"
#include "kernel/pad.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::pad(
    const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& arg0,
    const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& arg1, // scalar
    std::vector<std::shared_ptr<runtime::he::HECiphertext>>& out,
    const Shape& arg0_shape,
    const Shape& out_shape,
    const Shape& padding_below,
    const Shape& padding_above,
    const Shape& padding_interior,
    const std::shared_ptr<runtime::he::HEBackend>& he_backend)
{
    if (arg1.size() != 1)
    {
        throw ngraph_error("Padding element must be scalar");
    }

    // Todo: pad_val shall be arg1[0]. There's an unknown issue causing the computation
    //       to return -inf when arg1[0] is used. Luckily since we are doing mnist, the output
    //       values near the edge of the image are all zero in the first conv later, so it happens
    //       to pad zero in our case. This is not true for other models.
    std::shared_ptr<runtime::he::HECiphertext> pad_val = arg1[0];

    auto he_heaan_backend = dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(he_backend);
    std::shared_ptr<runtime::he::HEPlaintext> plaintext =
            dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(
                he_heaan_backend->create_empty_plaintext());

    he_heaan_backend->decrypt(plaintext, pad_val);

    float val=  dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(plaintext)->m_plaintexts[0];

    NGRAPH_INFO << "val " << val;

    Coordinate input_start(arg0_shape.size(), 0); // start at (0,0,...,0)
    Coordinate input_end =
        out_shape; // end at (d'0,d'1,...,d'n), the outer corner of the post-padding shape

    Strides input_strides(arg0_shape.size(), 1);

    AxisVector input_axis_order(arg0_shape.size());
    for (size_t i = 0; i < arg0_shape.size(); i++)
    {
        input_axis_order[i] = i;
    }

    Strides input_dilation(arg0_shape.size());
    for (size_t i = 0; i < arg0_shape.size(); i++)
    {
        input_dilation[i] = padding_interior[i] + 1;
    }

    // Need to cast these to CoordinateDiff in order to make CoordinateTransform happy.
    CoordinateDiff padding_below_signed;
    CoordinateDiff padding_above_signed;

    for (size_t i = 0; i < padding_below.size(); i++)
    {
        padding_below_signed.push_back(padding_below[i]);
        padding_above_signed.push_back(padding_above[i]);
    }

    CoordinateTransform input_transform(arg0_shape,
                                        input_start,
                                        input_end,
                                        input_strides,
                                        input_axis_order,
                                        padding_below_signed,
                                        padding_above_signed,
                                        input_dilation);
    CoordinateTransform output_transform(out_shape);

    CoordinateTransform::Iterator output_it = output_transform.begin();

    for (const Coordinate& in_coord : input_transform)
    {
        const Coordinate& out_coord = *output_it;

        std::shared_ptr<runtime::he::HECiphertext> v =
            input_transform.has_source_coordinate(in_coord) ? arg0[input_transform.index(in_coord)]
                                                            : pad_val;

        out[output_transform.index(out_coord)] = v;

        ++output_it;
    }
}

void runtime::he::kernel::pad(
    const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& arg0,
    const std::vector<std::shared_ptr<runtime::he::HEPlaintext>>& arg1, // scalar
    std::vector<std::shared_ptr<runtime::he::HECiphertext>>& out,
    const Shape& arg0_shape,
    const Shape& out_shape,
    const Shape& padding_below,
    const Shape& padding_above,
    const Shape& padding_interior,
    const std::shared_ptr<runtime::he::HEBackend>& he_backend)
{
    if (arg1.size() != 1)
    {
        throw ngraph_error("Padding element must be scalar");
    }

    std::shared_ptr<runtime::he::HECiphertext> arg1_encrypted;

    if (auto he_seal_backend =
            dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(he_backend))
    {
        std::shared_ptr<runtime::he::HECiphertext> ciphertext =
            dynamic_pointer_cast<runtime::he::SealCiphertextWrapper>(
                he_seal_backend->create_empty_ciphertext());
        he_seal_backend->encrypt(ciphertext, arg1[0]);
        arg1_encrypted = ciphertext;
    }
    else if (auto he_heaan_backend =
                 dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(he_backend))
    {
        std::shared_ptr<runtime::he::HECiphertext> ciphertext =
            dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(
                he_heaan_backend->create_empty_ciphertext());
        he_heaan_backend->encrypt(ciphertext, arg1[0]);
        arg1_encrypted = ciphertext;
    }
    else
    {
        throw ngraph_error("Result backend is neither SEAL nor HEAAN.");
    }

    std::vector<std::shared_ptr<runtime::he::HECiphertext>> arg1_encrypted_vector{arg1_encrypted};

    runtime::he::kernel::pad(arg0,
                             arg1_encrypted_vector,
                             out,
                             arg0_shape,
                             out_shape,
                             padding_below,
                             padding_above,
                             padding_interior,
                             he_backend);
}
