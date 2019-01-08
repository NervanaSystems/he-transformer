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

#include "kernel/add.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/add_seal.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void kernel::scalar_add(const HECiphertext* arg0, const HECiphertext* arg1,
                        shared_ptr<HECiphertext>& out,
                        const element::Type& element_type,
                        const HEBackend* he_backend) {
  if (auto he_seal_backend =
          dynamic_cast<const he_seal::HESealBackend*>(he_backend)) {
    auto arg0_seal = static_cast<const he_seal::SealCiphertextWrapper*>(arg0);
    auto arg1_seal = static_cast<const he_seal::SealCiphertextWrapper*>(arg1);
    auto out_seal = dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(out);

    if (arg0_seal && arg1_seal && out_seal) {
      he_seal::kernel::scalar_add(arg0_seal, arg1_seal, out_seal, element_type,
                                  he_seal_backend);
      out = dynamic_pointer_cast<HECiphertext>(out_seal);
    } else {
      throw ngraph_error(
          "Add backend is SEAL, but arguments or outputs are not "
          "SealCiphertextWrapper");
    }
  } else {
    throw ngraph_error("Add backend is not SEAL.");
  }
}

void kernel::scalar_add(const HEPlaintext* arg0, const HEPlaintext* arg1,
                        shared_ptr<HEPlaintext>& out,
                        const element::Type& element_type,
                        const HEBackend* he_backend) {
  if (auto he_seal_backend =
          dynamic_cast<const he_seal::HESealBackend*>(he_backend)) {
    auto arg0_seal = static_cast<const he_seal::SealPlaintextWrapper*>(arg0);
    auto arg1_seal = static_cast<const he_seal::SealPlaintextWrapper*>(arg1);
    auto out_seal = dynamic_pointer_cast<he_seal::SealPlaintextWrapper>(out);

    if (out_seal) {
      he_seal::kernel::scalar_add(arg0_seal, arg1_seal, out_seal, element_type,
                                  he_seal_backend);
      out = dynamic_pointer_cast<HEPlaintext>(out_seal);
    } else {
      throw ngraph_error(
          "Add backend is SEAL, but arguments or outputs are not "
          "SealPlaintextWrapper.");
    }
  } else {
    throw ngraph_error("Add backend is not SEAL.");
  }
}

void kernel::scalar_add(const HECiphertext* arg0, const HEPlaintext* arg1,
                        shared_ptr<HECiphertext>& out,
                        const element::Type& element_type,
                        const HEBackend* he_backend) {
  if (auto he_seal_backend =
          dynamic_cast<const he_seal::HESealBackend*>(he_backend)) {
    auto arg0_seal = static_cast<const he_seal::SealCiphertextWrapper*>(arg0);
    auto arg1_seal = static_cast<const he_seal::SealPlaintextWrapper*>(arg1);
    auto out_seal = dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(out);

    if (out_seal) {
      he_seal::kernel::scalar_add(arg0_seal, arg1_seal, out_seal, element_type,
                                  he_seal_backend);
      out = dynamic_pointer_cast<HECiphertext>(out_seal);
    } else {
      throw ngraph_error(
          "Add backend is SEAL, but arguments or outputs are not "
          "SealPlaintextWrapper.");
    }
  } else {
    throw ngraph_error("Add backend is not SEAL.");
  }
}

void kernel::scalar_add(const HEPlaintext* arg0, const HECiphertext* arg1,
                        shared_ptr<HECiphertext>& out,
                        const element::Type& element_type,
                        const HEBackend* he_backend) {
  scalar_add(arg1, arg0, out, element_type, he_backend);
}
