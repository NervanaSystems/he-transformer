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
#include <utility>

#include "he_backend.hpp"
#include "he_cipher_tensor_view.hpp"
#include "kernel/add.hpp"
#include "kernel/dot.hpp"
#include "kernel/multiply.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/type/element_type.hpp"

void ngraph::runtime::he::kernel::dot(const vector<shared_ptr<seal::Ciphertext>>& arg0,
                                      const vector<shared_ptr<seal::Ciphertext>>& arg1,
                                      vector<shared_ptr<seal::Ciphertext>>& out,
                                      const Shape& arg0_shape,
                                      const Shape& arg1_shape,
                                      const Shape& out_shape,
                                      size_t reduction_axes_count,
                                      const element::Type& type,
                                      shared_ptr<HEBackend> he_backend)
{
    dot_template(
        arg0, arg1, out, arg0_shape, arg1_shape, out_shape, reduction_axes_count, type, he_backend);
}

void ngraph::runtime::he::kernel::dot(const vector<shared_ptr<seal::Ciphertext>>& arg0,
                                      const vector<shared_ptr<seal::Plaintext>>& arg1,
                                      vector<shared_ptr<seal::Ciphertext>>& out,
                                      const Shape& arg0_shape,
                                      const Shape& arg1_shape,
                                      const Shape& out_shape,
                                      size_t reduction_axes_count,
                                      const element::Type& type,
                                      shared_ptr<HEBackend> he_backend)
{
    dot_template(
        arg0, arg1, out, arg0_shape, arg1_shape, out_shape, reduction_axes_count, type, he_backend);
}

void ngraph::runtime::he::kernel::dot(const vector<shared_ptr<seal::Plaintext>>& arg0,
                                      const vector<shared_ptr<seal::Ciphertext>>& arg1,
                                      vector<shared_ptr<seal::Ciphertext>>& out,
                                      const Shape& arg0_shape,
                                      const Shape& arg1_shape,
                                      const Shape& out_shape,
                                      size_t reduction_axes_count,
                                      const element::Type& type,
                                      shared_ptr<HEBackend> he_backend)
{
    dot_template(
        arg0, arg1, out, arg0_shape, arg1_shape, out_shape, reduction_axes_count, type, he_backend);
}
