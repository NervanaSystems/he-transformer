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

#include "seal/kernel/add_seal.hpp"
#include "seal/bfv/kernel/add_seal_bfv.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/ckks/kernel/add_seal_ckks.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void he_seal::kernel::scalar_add(
    const he_seal::SealCiphertextWrapper* arg0,
    const he_seal::SealCiphertextWrapper* arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  if (auto he_seal_ckks_backend =
          dynamic_cast<const he_seal::HESealCKKSBackend*>(he_seal_backend)) {
    he_seal::ckks::kernel::scalar_add_ckks(arg0, arg1, out, element_type,
                                           he_seal_ckks_backend);
  } else if (auto he_seal_bfv_backend =
                 dynamic_cast<const he_seal::HESealBFVBackend*>(
                     he_seal_backend)) {
    he_seal::bfv::kernel::scalar_add_bfv(arg0, arg1, out, element_type,
                                         he_seal_bfv_backend);
  } else {
    throw ngraph_error("HESealBackend is neither BFV nor CKKS");
  }
}

void he_seal::kernel::scalar_add(const runtime::he::HECiphertext* arg0,
                                 const runtime::he::HECiphertext* arg1,
                                 shared_ptr<runtime::he::HECiphertext>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealBackend* he_seal_backend,
                                 const seal::MemoryPoolHandle& pool) {
  auto arg0_seal = static_cast<const he_seal::SealCiphertextWrapper*>(arg0);
  auto arg1_seal = static_cast<const he_seal::SealCiphertextWrapper*>(arg1);
  auto out_seal = static_pointer_cast<he_seal::SealCiphertextWrapper>(out);
  he_seal::kernel::scalar_add(arg0_seal, arg1_seal, out_seal, element_type,
                              he_seal_backend, pool);
  out = static_pointer_cast<HECiphertext>(out_seal);
}

void he_seal::kernel::scalar_add(
    const he_seal::SealCiphertextWrapper* arg0,
    const he_seal::SealPlaintextWrapper* arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  const string type_name = element_type.c_type_string();

  bool add_zero = he_seal_backend->optimized_add();
  if (add_zero) {
    auto arg0_plaintext =
        static_pointer_cast<const he_seal::SealPlaintextWrapper>(
            he_seal_backend->get_valued_plaintext(0))
            ->m_plaintext;

    add_zero = (arg1->m_plaintext == arg0_plaintext);
  }

  if (add_zero && type_name == "float") {
    NGRAPH_INFO << "Optimized add by 0";
    out = make_shared<he_seal::SealCiphertextWrapper>(*arg0);
  } else {
    if (auto he_seal_ckks_backend =
            dynamic_cast<const he_seal::HESealCKKSBackend*>(he_seal_backend)) {
      he_seal::ckks::kernel::scalar_add_ckks(arg0, arg1, out, element_type,
                                             he_seal_ckks_backend);
    } else if (auto he_seal_bfv_backend =
                   dynamic_cast<const he_seal::HESealBFVBackend*>(
                       he_seal_backend)) {
      he_seal::bfv::kernel::scalar_add_bfv(arg0, arg1, out, element_type,
                                           he_seal_bfv_backend);
    } else {
      throw ngraph_error("HESealBackend is neither BFV nor CKKS");
    }
  }
}

void he_seal::kernel::scalar_add(const runtime::he::HECiphertext* arg0,
                                 const runtime::he::HEPlaintext* arg1,
                                 shared_ptr<runtime::he::HECiphertext>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealBackend* he_seal_backend,
                                 const seal::MemoryPoolHandle& pool) {
  auto arg0_seal = static_cast<const he_seal::SealCiphertextWrapper*>(arg0);
  auto arg1_seal = static_cast<const he_seal::SealPlaintextWrapper*>(arg1);
  auto out_seal = static_pointer_cast<he_seal::SealCiphertextWrapper>(out);
  he_seal::kernel::scalar_add(arg0_seal, arg1_seal, out_seal, element_type,
                              he_seal_backend, pool);
  out = static_pointer_cast<HECiphertext>(out_seal);
}

void he_seal::kernel::scalar_add(
    const he_seal::SealPlaintextWrapper* arg0,
    const he_seal::SealCiphertextWrapper* arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  he_seal::kernel::scalar_add(arg1, arg0, out, element_type, he_seal_backend);
}

void he_seal::kernel::scalar_add(const runtime::he::HEPlaintext* arg0,
                                 const runtime::he::HECiphertext* arg1,
                                 shared_ptr<runtime::he::HECiphertext>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealBackend* he_seal_backend,
                                 const seal::MemoryPoolHandle& pool) {
  he_seal::kernel::scalar_add(arg1, arg0, out, element_type, he_seal_backend,
                              pool);
}

void he_seal::kernel::scalar_add(const he_seal::SealPlaintextWrapper* arg0,
                                 const he_seal::SealPlaintextWrapper* arg1,
                                 shared_ptr<he_seal::SealPlaintextWrapper>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealBackend* he_seal_backend,
                                 const seal::MemoryPoolHandle& pool) {
  /*
shared_ptr<HEPlaintext> out_he = dynamic_pointer_cast<HEPlaintext>(out);
const string type_name = element_type.c_type_string();
if (type_name == "float") {
float x, y;
he_seal_backend->decode(&x, arg0, element_type);
he_seal_backend->decode(&y, arg1, element_type);
float r = x + y;
he_seal_backend->encode(out_he, &r, element_type);
} else {
throw ngraph_error("Unsupported element type " + type_name + " in add");
}
out = static_pointer_cast<he_seal::SealPlaintextWrapper>(out_he);
*/
}

void he_seal::kernel::scalar_add(const runtime::he::HEPlaintext* arg0,
                                 const runtime::he::HEPlaintext* arg1,
                                 shared_ptr<runtime::he::HEPlaintext>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealBackend* he_seal_backend,
                                 const seal::MemoryPoolHandle& pool) {
  auto arg0_seal = static_cast<const he_seal::SealPlaintextWrapper*>(arg0);
  auto arg1_seal = static_cast<const he_seal::SealPlaintextWrapper*>(arg1);
  auto out_seal = static_pointer_cast<he_seal::SealPlaintextWrapper>(out);
  he_seal::kernel::scalar_add(arg0_seal, arg1_seal, out_seal, element_type,
                              he_seal_backend, pool);
  out = static_pointer_cast<HEPlaintext>(out_seal);
}