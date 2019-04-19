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

void kernel::scalar_subtract(HECiphertext* arg0, HECiphertext* arg1,
                             shared_ptr<HECiphertext>& out,
                             const element::Type& type,
                             const HEBackend* he_backend) {
  if (auto he_seal_backend =
          dynamic_cast<const he_seal::HESealBackend*>(he_backend)) {
    auto arg0_seal = dynamic_cast<he_seal::SealCiphertextWrapper*>(arg0);
    auto arg1_seal = dynamic_cast<he_seal::SealCiphertextWrapper*>(arg1);
    auto out_seal = dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(out);

    if (arg0_seal && arg1_seal && out_seal) {
      he_seal::kernel::scalar_subtract(arg0_seal, arg1_seal, out_seal, type,
                                       he_seal_backend);
      out = dynamic_pointer_cast<HECiphertext>(out_seal);
    } else {
      throw ngraph_error(
          "subtract backend is SEAL, but arguments or outputs are not "
          "SealCiphertextWrapper");
    }
  } else {
    throw ngraph_error("subtract backend is not SEAL.");
  }
}

void kernel::scalar_subtract(HEPlaintext* arg0, HEPlaintext* arg1,
                             shared_ptr<HEPlaintext>& out,
                             const element::Type& type,
                             const HEBackend* he_backend) {
  if (auto he_seal_backend =
          dynamic_cast<const he_seal::HESealBackend*>(he_backend)) {
    auto arg0_seal = dynamic_cast<he_seal::SealPlaintextWrapper*>(arg0);
    auto arg1_seal = dynamic_cast<he_seal::SealPlaintextWrapper*>(arg1);
    auto out_seal = dynamic_pointer_cast<he_seal::SealPlaintextWrapper>(out);

    if (arg0_seal && arg1_seal && out_seal) {
      he_seal::kernel::scalar_subtract(arg0_seal, arg1_seal, out_seal, type,
                                       he_seal_backend);
      out = dynamic_pointer_cast<HEPlaintext>(out_seal);
    } else {
      throw ngraph_error(
          "subtract backend is SEAL, but arguments or outputs are not "
          "SealPlaintextWrapper");
    }
  } else {
    throw ngraph_error("subtract backend is not SEAL.");
  }
}

void kernel::scalar_subtract(HECiphertext* arg0, HEPlaintext* arg1,
                             shared_ptr<HECiphertext>& out,
                             const element::Type& type,
                             const HEBackend* he_backend) {
  NGRAPH_ASSERT(type == element::f32) << "Only type float32 supported";

  auto he_seal_backend =
      dynamic_cast<const he_seal::HESealBackend*>(he_backend);

  NGRAPH_ASSERT(he_seal_backend != nullptr) << "HEBackend is not HESealBackend";

  auto arg0_seal = dynamic_cast<he_seal::SealCiphertextWrapper*>(arg0);
  auto arg1_seal = dynamic_cast<he_seal::SealPlaintextWrapper*>(arg1);
  auto out_seal = dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(out);

  NGRAPH_ASSERT(arg0_seal != nullptr) << "arg0 is not Seal Ciphertext";
  NGRAPH_ASSERT(arg1_seal != nullptr) << "arg1 is not Seal Plaintext";
  NGRAPH_ASSERT(out_seal != nullptr) << "out is not Seal Ciphertext";

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
    out = dynamic_pointer_cast<HECiphertext>(out_seal);
  }
}

void kernel::scalar_subtract(HEPlaintext* arg0, HECiphertext* arg1,
                             shared_ptr<HECiphertext>& out,
                             const element::Type& type,
                             const HEBackend* he_backend) {
  scalar_negate(arg1, out, type, he_backend);
  scalar_add(arg0, out.get(), out, type, he_backend);
}
