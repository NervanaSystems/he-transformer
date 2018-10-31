//*****************************************************************************
// Copyright 2018 Intel Corporation
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

#include "kernel/slice.hpp"
#include "he_backend.hpp"
#include "he_ciphertext.hpp"
#include "he_plaintext.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::slice(const vector<shared_ptr<runtime::he::HECiphertext>>& arg,
                                vector<shared_ptr<runtime::he::HECiphertext>>& out,
                                const Shape& arg_shape,
                                const Coordinate& lower_bounds,
                                const Coordinate& upper_bounds,
                                const Strides& strides,
                                const Shape& out_shape)
{
    slice<runtime::he::HECiphertext, runtime::he::HECiphertext>(
        arg, out, arg_shape, lower_bounds, upper_bounds, strides, out_shape);
}

void runtime::he::kernel::slice(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg,
                                vector<shared_ptr<runtime::he::HEPlaintext>>& out,
                                const Shape& arg_shape,
                                const Coordinate& lower_bounds,
                                const Coordinate& upper_bounds,
                                const Strides& strides,
                                const Shape& out_shape)
{
    slice<runtime::he::HEPlaintext, runtime::he::HEPlaintext>(
        arg, out, arg_shape, lower_bounds, upper_bounds, strides, out_shape);
}
