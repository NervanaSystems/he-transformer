//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
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

#include "kernel/pad.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/except.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::pad(
    const vector<shared_ptr<runtime::he::HECiphertext>>& arg0,
    const vector<shared_ptr<runtime::he::HEPlaintext>>& arg1,  // scalar
    vector<shared_ptr<runtime::he::HECiphertext>>& out, const Shape& arg0_shape,
    const Shape& out_shape, const Shape& padding_below,
    const Shape& padding_above, const Shape& padding_interior,
    size_t batch_size, const runtime::he::HEBackend* he_backend) {
  if (arg1.size() != 1) {
    throw ngraph_error("Padding element must be scalar");
  }

  shared_ptr<runtime::he::HECiphertext> arg1_encrypted =
      he_backend->create_empty_ciphertext();
  he_backend->encrypt(arg1_encrypted, *arg1[0]);

  vector<shared_ptr<runtime::he::HECiphertext>> arg1_encrypted_vector{
      arg1_encrypted};

  runtime::he::kernel::pad(arg0, arg1_encrypted_vector, out, arg0_shape,
                           out_shape, padding_below, padding_above,
                           padding_interior, batch_size, he_backend);
}