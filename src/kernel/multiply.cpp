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

#include <vector>

#include "kernel/multiply.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/multiply_seal.hpp"
#include "seal/kernel/negate_seal.hpp"

void kernel::scalar_multiply(std::shared_ptr<ngraph::he::HECiphertext>& arg0,
                             std::shared_ptr<ngraph::he::HECiphertext>& arg1,
                             std::shared_ptr<ngraph::he::HECiphertext>& out,
                             const element::Type& element_type,
                             const HEBackend* he_backend) {
  auto he_seal_backend = cast_to_seal_backend(he_backend);
  auto arg0_seal = cast_to_seal_hetext(arg0);
  auto arg1_seal = cast_to_seal_hetext(arg1);
  auto out_seal = cast_to_seal_hetext(out);
  kernel::scalar_multiply(arg0_seal, arg1_seal, out_seal, element_type,
                          he_seal_backend);
  out = dynamic_pointer_cast<ngraph::he::HECiphertext>(out_seal);
}

void kernel::scalar_multiply(std::shared_ptr<HEPlaintext>& arg0,
                             std::shared_ptr<HEPlaintext>& arg1,
                             std::shared_ptr<HEPlaintext>& out,
                             const element::Type& element_type,
                             const HEBackend* he_backend) {
  auto he_seal_backend = cast_to_seal_backend(he_backend);
  auto arg0_seal = cast_to_seal_hetext(arg0);
  auto arg1_seal = cast_to_seal_hetext(arg1);
  auto out_seal = cast_to_seal_hetext(out);
  kernel::scalar_multiply(arg0_seal, arg1_seal, out_seal, element_type,
                          he_seal_backend);
  out = dynamic_pointer_cast<HEPlaintext>(out_seal);
}

void kernel::scalar_multiply(std::shared_ptr<ngraph::he::HECiphertext>& arg0,
                             std::shared_ptr<HEPlaintext>& arg1,
                             std::shared_ptr<ngraph::he::HECiphertext>& out,
                             const element::Type& element_type,
                             const HEBackend* he_backend) {
  auto he_seal_backend = cast_to_seal_backend(he_backend);
  auto arg0_seal = cast_to_seal_hetext(arg0);
  auto arg1_seal = cast_to_seal_hetext(arg1);
  auto out_seal = cast_to_seal_hetext(out);

  kernel::scalar_multiply(arg0_seal, arg1_seal, out_seal, element_type,
                          he_seal_backend);
  out = dynamic_pointer_cast<ngraph::he::HECiphertext>(out_seal);
}

void kernel::scalar_multiply(std::shared_ptr<HEPlaintext>& arg0,
                             std::shared_ptr<ngraph::he::HECiphertext>& arg1,
                             std::shared_ptr<ngraph::he::HECiphertext>& out,
                             const element::Type& element_type,
                             const ngraph::he::HEBackend* he_backend) {
  scalar_multiply(arg1, arg0, out, element_type, he_backend);
}
