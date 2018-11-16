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

#include "seal/kernel/multiply_seal.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/bfv/kernel/multiply_seal_bfv.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/ckks/kernel/multiply_seal_ckks.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void he_seal::kernel::scalar_multiply(
    const he_seal::SealCiphertextWrapper* arg0,
    const he_seal::SealCiphertextWrapper* arg1,
    he_seal::SealCiphertextWrapper* out, const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend) {
  if (auto he_seal_ckks_backend =
          dynamic_cast<const he_seal::HESealCKKSBackend*>(he_seal_backend)) {
    he_seal::ckks::kernel::scalar_multiply_ckks(arg0, arg1, out, element_type,
                                                he_seal_ckks_backend);
  } else if (auto he_seal_bfv_backend =
                 dynamic_cast<const he_seal::HESealBFVBackend*>(
                     he_seal_backend)) {
    he_seal::bfv::kernel::scalar_multiply_bfv(arg0, arg1, out, element_type,
                                              he_seal_bfv_backend);
  } else {
    throw ngraph_error("HESealBackend is neither BFV nor CKKS");
  }
}

void he_seal::kernel::scalar_multiply(
    const he_seal::SealCiphertextWrapper* arg0,
    const he_seal::SealPlaintextWrapper* arg1,
    he_seal::SealCiphertextWrapper* out, const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend) {
  if (auto he_seal_ckks_backend =
          dynamic_cast<const he_seal::HESealCKKSBackend*>(he_seal_backend)) {
    he_seal::ckks::kernel::scalar_multiply_ckks(arg0, arg1, out, element_type,
                                                he_seal_ckks_backend);
  } else if (auto he_seal_bfv_backend =
                 dynamic_cast<const he_seal::HESealBFVBackend*>(
                     he_seal_backend)) {
    he_seal::bfv::kernel::scalar_multiply_bfv(arg0, arg1, out, element_type,
                                              he_seal_bfv_backend);
  } else {
    throw ngraph_error("HESealBackend is neither BFV nor CKKS");
  }
}

void he_seal::kernel::scalar_multiply(
    const he_seal::SealPlaintextWrapper* arg0,
    const he_seal::SealCiphertextWrapper* arg1,
    he_seal::SealCiphertextWrapper* out, const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend) {
  he_seal::kernel::scalar_multiply(arg1, arg0, out, element_type,
                                   he_seal_backend);
}

void he_seal::kernel::scalar_multiply(
    const he_seal::SealPlaintextWrapper* arg0,
    const he_seal::SealPlaintextWrapper* arg1,
    he_seal::SealPlaintextWrapper* out, const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend) {
  throw ngraph_error("Scalar multiply P*P not enabled");
  /*
auto out_he = dynamic_cast<runtime::he::HEPlaintext*>(out);
const string type_name = element_type.c_type_string();
if (type_name == "float") {
float x, y;
he_seal_backend->decode(&x, arg0, element_type);
he_seal_backend->decode(&y, arg1, element_type);
float r = x * y;
he_seal_backend->encode(out_he, &r, element_type);
} else {
throw ngraph_error("Unsupported element type " + type_name +
                   " in multiply");
}
out = dynamic_cast<runtime::he::he_seal::SealPlaintextWrapper*>(out_he); */
}