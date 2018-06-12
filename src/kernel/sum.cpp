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
#include "he_cipher_tensor_view.hpp"
#include "he_ciphertext.hpp"
#include "he_heaan_backend.hpp"
#include "he_plain_tensor_view.hpp"
#include "he_seal_backend.hpp"
#include "kernel/add.hpp"
#include "kernel/sum.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::sum(const vector<shared_ptr<runtime::he::HECiphertext>>& arg,
                              vector<shared_ptr<runtime::he::HECiphertext>>& out,
                              const Shape& in_shape,
                              const Shape& out_shape,
                              const AxisSet& reduction_axes,
                              const element::Type& type,
                              const shared_ptr<runtime::he::HEBackend>& he_backend)
{
    auto he_seal_backend = dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(he_backend);
    auto he_heaan_backend = dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(he_backend);
    if (!he_seal_backend && !he_heaan_backend)
    {
        throw ngraph_error("Sum backend is neither SEAL nor HEAAN.);
    }

    CoordinateTransform output_transform(out_shape);

    shared_ptr<HECipherTensorView> zero_tv;

    if (he_seal_backend != nullptr)
    {
        zero_tv = static_pointer_cast<HECipherTensorView>(
            he_seal_backend->create_valued_tensor(0., type, out_shape));
    }
    else if (he_heaan_backend != nullptr)
    {
        zero_tv = static_pointer_cast<HECipherTensorView>(
            he_heaan_backend->create_valued_tensor(0., type, out_shape));
    }

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
        size_t output_ind = output_transform.index(output_coord);

        shared_ptr<runtime::he::HECiphertext> cipher_out = out[output_ind];

        runtime::he::kernel::scalar_add(
            cipher_out, arg[input_transform.index(input_coord)], cipher_out, type, he_backend);
    }
}

void runtime::he::kernel::sum(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg,
                              vector<shared_ptr<runtime::he::HEPlaintext>>& out,
                              const Shape& in_shape,
                              const Shape& out_shape,
                              const AxisSet& reduction_axes,
                              const element::Type& type,
                              const shared_ptr<runtime::he::HEBackend>& he_backend)
{
    auto he_seal_backend = dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(he_backend);
    auto he_heaan_backend = dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(he_backend);
    if (!he_seal_backend && !he_heaan_backend)
    {
        throw ngraph_error("Sum backend is neither SEAL nor HEAAN.);
    }

    CoordinateTransform output_transform(out_shape);

    shared_ptr<HEPlainTensorView> zero_tv;

    if (he_seal_backend != nullptr)
    {
        zero_tv = static_pointer_cast<HEPlainTensorView>(
            he_seal_backend->create_valued_plain_tensor(0., type, out_shape));
    }
    else if (he_heaan_backend != nullptr)
    {
        zero_tv = static_pointer_cast<HEPlainTensorView>(
            he_heaan_backend->create_valued_plain_tensor(0., type, out_shape));
    }

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
        size_t output_ind = output_transform.index(output_coord);

        shared_ptr<runtime::he::HEPlaintext> plain_out = out[output_ind];

        runtime::he::kernel::scalar_add(
            plain_out, arg[input_transform.index(input_coord)], plain_out, type, he_backend);

        out[output_ind] = plain_out;
    }
}
