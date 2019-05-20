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
    shared_ptr<he_seal::SealCiphertextWrapper>& arg0,
    shared_ptr<he_seal::SealCiphertextWrapper>& arg1,
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
    shared_ptr<he_seal::SealCiphertextWrapper>& arg0,
    shared_ptr<he_seal::SealPlaintextWrapper>& arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  NGRAPH_ASSERT(element_type == element::f32)
      << "Element type " << element_type << " is not float";

  const auto& values = arg1->get_values();
  // TODO: check multiplying by small numbers behavior more thoroughly
  if (std::all_of(values.begin(), values.end(),
                  [](float f) { return std::abs(f) < 1e-5f; })) {
    out = dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(
        he_seal_backend->create_valued_ciphertext(0, element_type));
  } else if (std::all_of(values.begin(), values.end(),
                         [](float f) { return f == 1.0f; })) {
    // TODO: make copy only if needed
    NGRAPH_INFO << "Optimized mult by 1";
    out = make_shared<he_seal::SealCiphertextWrapper>(*arg0);
  } else if (std::all_of(values.begin(), values.end(),
                         [](float f) { return f == -1.0f; })) {
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
    shared_ptr<he_seal::SealPlaintextWrapper>& arg0,
    shared_ptr<he_seal::SealCiphertextWrapper>& arg1,
    shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  he_seal::kernel::scalar_multiply(arg1, arg0, out, element_type,
                                   he_seal_backend, pool);
}

void he_seal::kernel::scalar_multiply(
    shared_ptr<he_seal::SealPlaintextWrapper>& arg0,
    shared_ptr<he_seal::SealPlaintextWrapper>& arg1,
    shared_ptr<he_seal::SealPlaintextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  NGRAPH_ASSERT(element_type == element::f32);

  std::vector<float> arg0_vals = arg0->get_values();
  std::vector<float> arg1_vals = arg1->get_values();
  std::vector<float> out_vals(arg0->num_values());

  NGRAPH_ASSERT(arg0_vals.size() > 0)
      << "Multiplying plaintext arg0 has 0 values";
  NGRAPH_ASSERT(arg1_vals.size() > 0)
      << "Multiplying plaintext arg1 has 0 values";

  if (arg0_vals.size() == 1) {
    std::transform(arg1_vals.begin(), arg1_vals.end(), out_vals.begin(),
                   std::bind(std::multiplies<float>(), std::placeholders::_1,
                             arg0_vals[0]));
  } else if (arg1_vals.size() == 1) {
    std::transform(arg0_vals.begin(), arg0_vals.end(), out_vals.begin(),
                   std::bind(std::multiplies<float>(), std::placeholders::_1,
                             arg1_vals[0]));
  } else {
    std::transform(arg0_vals.begin(), arg0_vals.end(), arg1_vals.begin(),
                   out_vals.begin(), std::multiplies<float>());
  }
  out->set_values(out_vals);
}
