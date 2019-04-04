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
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"

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

  auto he_seal_backend =
      dynamic_cast<const runtime::he::he_seal::HESealBackend*>(he_backend);

  if (he_seal_backend == nullptr) {
    throw ngraph_error("Pad supports only SEAL backend");
  }

  shared_ptr<runtime::he::HECiphertext> arg1_encrypted;
  arg1_encrypted = he_seal_backend->create_empty_ciphertext();

  he_backend->encrypt(arg1_encrypted, *arg1[0]);

  // Change output modulus to match other ciphertexts in vector
  if (arg0.size() > 0) {
    auto arg0_wrapper =
        dynamic_pointer_cast<runtime::he::he_seal::SealCiphertextWrapper>(
            arg0[0]);
    assert(arg0_wrapper != nullptr);
    size_t chain_ind0 =
        he_seal_backend->get_context()
            ->context_data(arg0_wrapper->get_hetext().parms_id())
            ->chain_index();

    auto arg1_wrapper =
        dynamic_pointer_cast<runtime::he::he_seal::SealCiphertextWrapper>(
            arg1_encrypted);
    assert(arg1_wrapper != nullptr);

    size_t chain_ind_out =
        he_seal_backend->get_context()
            ->context_data(arg1_wrapper->get_hetext().parms_id())
            ->chain_index();

    NGRAPH_ASSERT(chain_ind_out >= chain_ind0)
        << "Encrypted pad value has smaller chain index that input";

    if (chain_ind_out > chain_ind0) {
      he_seal_backend->get_evaluator()->mod_switch_to_inplace(
          arg1_wrapper->get_hetext(), arg0_wrapper->get_hetext().parms_id());
      chain_ind_out = he_seal_backend->get_context()
                          ->context_data(arg1_wrapper->get_hetext().parms_id())
                          ->chain_index();
      assert(chain_ind_out == chain_ind0);

      arg1_wrapper->get_hetext().scale() = arg0_wrapper->get_hetext().scale();
    }
  }

  vector<shared_ptr<runtime::he::HECiphertext>> arg1_encrypted_vector{
      arg1_encrypted};

  runtime::he::kernel::pad(arg0, arg1_encrypted_vector, out, arg0_shape,
                           out_shape, padding_below, padding_above,
                           padding_interior, batch_size, he_backend);
}