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
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/ckks/seal_ckks_util.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"

void ngraph::he::pad(
    std::vector<std::shared_ptr<ngraph::he::HECiphertext>>& arg0,
    std::vector<std::shared_ptr<ngraph::he::HEPlaintext>>& arg1,  // scalar
    std::vector<std::shared_ptr<ngraph::he::HECiphertext>>& out,
    const Shape& arg0_shape, const Shape& out_shape,
    const CoordinateDiff& padding_below, const CoordinateDiff& padding_above,
    op::PadMode pad_mode, size_t batch_size,
    const ngraph::he::HESealBackend* he_seal_backend) {
  if (arg1.size() != 1) {
    throw ngraph_error("Padding element must be scalar");
  }

  auto he_seal_backend = ngraph::he::cast_to_seal_backend(he_seal_backend);

  std::shared_ptr<ngraph::he::HECiphertext> arg1_encrypted;
  arg1_encrypted = he_seal_backend->create_empty_ciphertext();

  bool is_pad_value_zero =
      arg1[0]->is_single_value() && arg1[0]->get_values()[0] == 0.;
  NGRAPH_CHECK(is_pad_value_zero, "Non-zero pad values not supported");
  arg1_encrypted->set_zero(true);

  std::vector<std::shared_ptr<ngraph::he::HECiphertext>> arg1_encrypted_vector{
      arg1_encrypted};

  ngraph::he::pad(arg0, arg1_encrypted_vector, out, arg0_shape, out_shape,
                  padding_below, padding_above, pad_mode, batch_size,
                  he_seal_backend);
}
