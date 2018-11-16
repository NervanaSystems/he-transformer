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

#include <vector>

#include "kernel/negate.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/negate_seal.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void kernel::scalar_negate(const HECiphertext* arg, HECiphertext* out,
                           const element::Type& element_type,
                           const HEBackend* he_backend) {
  if (auto he_seal_backend =
          dynamic_cast<const he_seal::HESealBackend*>(he_backend)) {
    auto arg_seal = dynamic_cast<const he_seal::SealCiphertextWrapper*>(arg);
    auto out_seal = dynamic_cast<he_seal::SealCiphertextWrapper*>(out);

    if (arg_seal && out_seal) {
      he_seal::kernel::scalar_negate(arg_seal, out_seal, element_type,
                                     he_seal_backend);
      out = dynamic_cast<HECiphertext*>(out_seal);
    } else {
      throw ngraph_error(
          "negate backend is SEAL, but arguments or outputs are not "
          "SealCiphertextWrapper");
    }
  } else {
    throw ngraph_error("negate backend is not SEAL");
  }
}

void kernel::scalar_negate(const HEPlaintext* arg, HEPlaintext* out,
                           const element::Type& element_type,
                           const HEBackend* he_backend) {
  if (auto he_seal_backend =
          dynamic_cast<const he_seal::HESealBackend*>(he_backend)) {
    auto arg_seal = dynamic_cast<const he_seal::SealPlaintextWrapper*>(arg);
    auto out_seal = dynamic_cast<he_seal::SealPlaintextWrapper*>(out);

    if (arg_seal && out_seal) {
      he_seal::kernel::scalar_negate(arg_seal, out_seal, element_type,
                                     he_seal_backend);
      out = dynamic_cast<HEPlaintext*>(out_seal);
    } else {
      throw ngraph_error(
          "negate backend is SEAL, but arguments or outputs are not "
          "SealPlaintextWrapper");
    }
  } else {
    throw ngraph_error("negate backend is not SEAL");
  }
}