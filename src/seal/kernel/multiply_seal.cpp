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

#include "seal/kernel/multiply_seal.hpp"
#include "seal/bfv/kernel/multiply_seal_bfv.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/ckks/kernel/multiply_seal_ckks.hpp"
#include "seal/kernel/negate_seal.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void he_seal::kernel::scalar_multiply(
    he_seal::SealCiphertextWrapper* arg0, he_seal::SealCiphertextWrapper* arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  if (auto he_seal_ckks_backend =
          dynamic_cast<const he_seal::HESealCKKSBackend*>(he_seal_backend)) {
    he_seal::ckks::kernel::scalar_multiply_ckks(arg0, arg1, out, element_type,
                                                he_seal_ckks_backend, pool);
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
    HECiphertext* arg0, HECiphertext* arg1, shared_ptr<HECiphertext>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  auto arg0_seal = static_cast<he_seal::SealCiphertextWrapper*>(arg0);
  auto arg1_seal = static_cast<he_seal::SealCiphertextWrapper*>(arg1);
  auto out_seal = static_pointer_cast<he_seal::SealCiphertextWrapper>(out);
  he_seal::kernel::scalar_multiply(arg0_seal, arg1_seal, out_seal, element_type,
                                   he_seal_backend, pool);
  out = static_pointer_cast<HECiphertext>(out_seal);
}

void he_seal::kernel::scalar_multiply(
    he_seal::SealCiphertextWrapper* arg0, he_seal::SealPlaintextWrapper* arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  enum class Optimization {
    mult_zero,
    mult_one,
    mult_neg_one,
    no_optimization
  };
  Optimization optimization = Optimization::no_optimization;

  NGRAPH_ASSERT(element_type == element::f32)
      << "Element type " << element_type << " is not float";

  if (he_seal_backend->optimized_mult()) {
    // TODO: less hacky way of checking for 0 in case of small floating-point
    // weights
    if (arg1->is_single_value() && arg1->get_value() == 0.0f) {
      optimization = Optimization::mult_zero;
    } else if (arg1->is_single_value() && arg1->get_value() == 1.0f) {
      optimization = Optimization::mult_one;
    } else if (arg1->is_single_value() && arg1->get_value() == -1.0f) {
      optimization = Optimization::mult_neg_one;
    }
  }

  if (optimization == Optimization::mult_zero) {
    out = dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(
        he_seal_backend->create_valued_ciphertext(0, element_type));
  } else if (optimization == Optimization::mult_one) {
    out = make_shared<he_seal::SealCiphertextWrapper>(*arg0);
  } else if (optimization == Optimization::mult_neg_one) {
    he_seal::kernel::scalar_negate(arg0, out, element_type, he_seal_backend);
  } else {
    if (auto he_seal_ckks_backend =
            dynamic_cast<const he_seal::HESealCKKSBackend*>(he_seal_backend)) {
      he_seal::ckks::kernel::scalar_multiply_ckks(arg0, arg1, out, element_type,
                                                  he_seal_ckks_backend, pool);
    } else if (auto he_seal_bfv_backend =
                   dynamic_cast<const he_seal::HESealBFVBackend*>(
                       he_seal_backend)) {
      he_seal::bfv::kernel::scalar_multiply_bfv(arg0, arg1, out, element_type,
                                                he_seal_bfv_backend);
    } else {
      throw ngraph_error("HESealBackend is neither BFV nor CKKS");
    }
  }
}

void he_seal::kernel::scalar_multiply(
    HECiphertext* arg0, HEPlaintext* arg1, shared_ptr<HECiphertext>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  auto arg0_seal = static_cast<he_seal::SealCiphertextWrapper*>(arg0);
  auto arg1_seal = static_cast<he_seal::SealPlaintextWrapper*>(arg1);
  auto out_seal = static_pointer_cast<he_seal::SealCiphertextWrapper>(out);
  he_seal::kernel::scalar_multiply(arg0_seal, arg1_seal, out_seal, element_type,
                                   he_seal_backend, pool);
  out = static_pointer_cast<HECiphertext>(out_seal);
}

void he_seal::kernel::scalar_multiply(
    HEPlaintext* arg0, HECiphertext* arg1, shared_ptr<HECiphertext>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  he_seal::kernel::scalar_multiply(arg1, arg0, out, element_type,
                                   he_seal_backend, pool);
}

void he_seal::kernel::scalar_multiply(
    he_seal::SealPlaintextWrapper* arg0, he_seal::SealCiphertextWrapper* arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  he_seal::kernel::scalar_multiply(arg1, arg0, out, element_type,
                                   he_seal_backend, pool);
}

void he_seal::kernel::scalar_multiply(
    he_seal::SealPlaintextWrapper* arg0, he_seal::SealPlaintextWrapper* arg1,
    shared_ptr<he_seal::SealPlaintextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  // TODO: generalize to multiple batch sizes
  shared_ptr<runtime::he::HEPlaintext> out_he =
      dynamic_pointer_cast<runtime::he::HEPlaintext>(out);
  NGRAPH_ASSERT(element_type == element::f32)
      << "Element type " << element_type << " is not float";

  float x, y;
  he_seal_backend->decode(&x, arg0, element_type);
  he_seal_backend->decode(&y, arg1, element_type);
  float r = x * y;
  he_seal_backend->encode(out_he, &r, element_type);
  out = static_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(out_he);
}

void he_seal::kernel::scalar_multiply(
    HEPlaintext* arg0, HEPlaintext* arg1, shared_ptr<HEPlaintext>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  auto arg0_seal = static_cast<he_seal::SealPlaintextWrapper*>(arg0);
  auto arg1_seal = static_cast<he_seal::SealPlaintextWrapper*>(arg1);
  auto out_seal = static_pointer_cast<he_seal::SealPlaintextWrapper>(out);
  he_seal::kernel::scalar_multiply(arg0_seal, arg1_seal, out_seal, element_type,
                                   he_seal_backend, pool);
  out = static_pointer_cast<HEPlaintext>(out_seal);
}