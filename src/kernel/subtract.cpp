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

#include "kernel/add.hpp"
#include "kernel/negate.hpp"
#include "kernel/subtract.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/subtract_seal.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void kernel::scalar_subtract(shared_ptr<HECiphertext>& arg0,
                             shared_ptr<HECiphertext>& arg1,
                             shared_ptr<HECiphertext>& out,
                             const element::Type& element_type,
                             const HEBackend* he_backend) {
  auto he_seal_backend = he_seal::cast_to_seal_backend(he_backend);
  auto arg0_seal = he_seal::cast_to_seal_hetext(arg0);
  auto arg1_seal = he_seal::cast_to_seal_hetext(arg1);
  auto out_seal = he_seal::cast_to_seal_hetext(out);

  he_seal::kernel::scalar_subtract(arg0_seal, arg1_seal, out_seal, element_type,
                                   he_seal_backend);
  out = dynamic_pointer_cast<HECiphertext>(out_seal);
}

void kernel::scalar_subtract(shared_ptr<HEPlaintext>& arg0,
                             shared_ptr<HEPlaintext>& arg1,
                             shared_ptr<HEPlaintext>& out,
                             const element::Type& element_type,
                             const HEBackend* he_backend) {
  NGRAPH_ASSERT(element_type == element::f32);

  std::vector<float> arg0_vals = arg0->get_values();
  std::vector<float> arg1_vals = arg1->get_values();
  std::vector<float> out_vals(arg0->num_values());

  std::transform(arg0_vals.begin(), arg0_vals.end(), arg1_vals.begin(),
                 out_vals.begin(), std::minus<float>());

  out->set_values(out_vals);
}

void kernel::scalar_subtract(shared_ptr<HECiphertext>& arg0,
                             shared_ptr<HEPlaintext>& arg1,
                             shared_ptr<HECiphertext>& out,
                             const element::Type& type,
                             const HEBackend* he_backend) {
  NGRAPH_ASSERT(type == element::f32) << "Only type float32 supported";

  auto he_seal_backend = he_seal::cast_to_seal_backend(he_backend);
  auto arg0_seal = he_seal::cast_to_seal_hetext(arg0);
  auto arg1_seal = he_seal::cast_to_seal_hetext(arg1);
  auto out_seal = he_seal::cast_to_seal_hetext(out);

  bool sub_zero =
      arg1_seal->is_single_value() && (arg1_seal->get_values()[0] == 0.0f);

  if (sub_zero) {
    // Make copy of input
    // TODO: make copy only if necessary
    NGRAPH_INFO << "Sub 0 optimization";
    out = static_pointer_cast<HECiphertext>(
        make_shared<he_seal::SealCiphertextWrapper>(*arg0_seal));
  } else {
    he_seal::kernel::scalar_subtract(arg0_seal, arg1_seal, out_seal, type,
                                     he_seal_backend);
  }
}

void kernel::scalar_subtract(shared_ptr<HEPlaintext>& arg0,
                             shared_ptr<HECiphertext>& arg1,
                             shared_ptr<HECiphertext>& out,
                             const element::Type& type,
                             const HEBackend* he_backend) {
  if (arg1->is_zero()) {
    he_backend->encrypt(out, arg0);
  } else {
    scalar_negate(arg1, out, type, he_backend);
    scalar_add(arg0, out, out, type, he_backend);
  }
}
