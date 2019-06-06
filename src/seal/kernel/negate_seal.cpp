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

#include "seal/kernel/negate_seal.hpp"

void ngraph::he::scalar_negate_seal(
    const ngraph::he::SealCiphertextWrapper& arg,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(element_type == element::f32);

  if (arg.is_zero()) {
    out->is_zero() = true;
    return;
  }
  he_seal_backend.get_evaluator()->negate(arg.ciphertext(), out->ciphertext());
}

void ngraph::he::scalar_negate_seal(const HEPlaintext& arg, HEPlaintext& out,
                                    const element::Type& element_type) {
  const std::vector<float>& arg_vals = arg.values();
  std::vector<float> out_vals(arg.num_values());

  std::transform(arg_vals.begin(), arg_vals.end(), out_vals.begin(),
                 std::negate<float>());
  out.values() = out_vals;
}
