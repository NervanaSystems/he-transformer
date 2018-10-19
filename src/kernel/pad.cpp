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
    size_t batch_size,
    const std::shared_ptr<runtime::he::HEBackend>& he_backend)
{
    if (arg1.size() != 1)
    {
        throw ngraph_error("Padding element must be scalar");
    }

    NGRAPH_INFO << "pad cipher ciper";

    std::shared_ptr<runtime::he::HECiphertext> pad_val = arg1[0];

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
        NGRAPH_INFO << "Set to pad val";

        out[output_transform.index(out_coord)] = v;

        ++output_it;
    }
    NGRAPH_INFO << "Done padding";
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
    size_t batch_size,
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
        if (arg0.size() == 0)
        {
            std::shared_ptr<runtime::he::HECiphertext> ciphertext =
                dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(
                    he_heaan_backend->create_empty_ciphertext());
            he_heaan_backend->encrypt(ciphertext, arg1[0]);
            arg1_encrypted = ciphertext;
        }
        else // Ensure arg0 and arg1 has the same precision and logq.
        {
            // TODO: move into he_backend
            auto arg0_heaan = dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(arg0[0]);

            if (batch_size == 1)
            {
                heaan::Ciphertext ciphertext = he_heaan_backend->get_scheme()->encryptSingle(
                                    dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(arg1[0])->m_plaintexts[0],
                                    arg0_heaan->m_ciphertext.logp,
                                    arg0_heaan->m_ciphertext.logq);
                NGRAPH_INFO << "Padding with zeros size " << arg0_heaan->m_count;
                arg1_encrypted = make_shared<runtime::he::HeaanCiphertextWrapper>(ciphertext, arg0_heaan->m_count);
            }
            else
            {
                double pad_value = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(arg1[0])->m_plaintexts[0];
                NGRAPH_INFO << "pad value " << pad_value;
                vector<double> plaintexts(batch_size, pad_value);

                NGRAPH_INFO << "Padding with zeros batch size " << batch_size;
                NGRAPH_INFO << "plaintext size " << plaintexts.size();
                for (auto elem : plaintexts)
                {
                    NGRAPH_INFO << "elem " << elem;
                }

               heaan::Ciphertext ciphertext = he_heaan_backend->get_scheme()->encrypt(
                    plaintexts,
                    arg0_heaan->m_ciphertext.logp,
                    arg0_heaan->m_ciphertext.logq);

                arg1_encrypted = make_shared<runtime::he::HeaanCiphertextWrapper>(ciphertext, batch_size);
            }
        }
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
                             batch_size,
                             he_backend);
}