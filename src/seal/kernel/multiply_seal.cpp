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
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/ckks/kernel/multiply_seal_ckks.hpp"
#include "seal/kernel/negate_seal.hpp"

void ngraph::he::scalar_multiply(
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg0,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  if (arg0->is_zero()) {
    out->set_zero(true);
  } else if (arg1->is_zero()) {
    out->set_zero(true);
  } else {
    auto he_seal_ckks_backend =
        ngraph::he::cast_to_seal_ckks_backend(he_seal_backend);
    ngraph::he::scalar_multiply_ckks(arg0, arg1, out, element_type,
                                     he_seal_ckks_backend, pool);
  }
}

void ngraph::he::scalar_multiply(
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg0,
    std::shared_ptr<ngraph::he::HEPlaintext>& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  NGRAPH_CHECK(element_type == element::f32, "Element type ", element_type,
               " is not float");
  if (arg0->is_zero()) {
    out->set_zero(true);
    return;
  }

  const auto& values = arg1->get_values();
  // TODO: check multiplying by small numbers behavior more thoroughly
  if (std::all_of(values.begin(), values.end(),
                  [](float f) { return std::abs(f) < 1e-5f; })) {
    out->set_zero(true);
    out = std::dynamic_pointer_cast<ngraph::he::SealCiphertextWrapper>(
        he_seal_backend->create_valued_ciphertext(0, element_type));
  }

  // We can't just do these scalar +/-1 optimizations, unless all the weights
  // are +/-1 in this layer, since we expect the scale of the ciphertext to
  // square. For instance, if we are computing c1*p(1) + c2 *p(2), the latter
  // sum will have larger scale than the former
  /*else if (std::all_of(values.begin(), values.end(),
                         [](float f) { return f == 1.0f; })) {
    out = make_shared<ngraph::he::SealCiphertextWrapper>(*arg0);
  } else if (std::all_of(values.begin(), values.end(),
                         [](float f) { return f == -1.0f; })) {
    scalar_negate(arg0, out, element_type, he_seal_backend);
  } */
  else {
    auto he_seal_ckks_backend =
        ngraph::he::cast_to_seal_ckks_backend(he_seal_backend);
    ngraph::he::scalar_multiply_ckks(arg0, arg1, out, element_type,
                                     he_seal_ckks_backend, pool);
  }
}

void ngraph::he::scalar_multiply(
    std::shared_ptr<ngraph::he::HEPlaintext>& arg0,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  ngraph::he::scalar_multiply(arg1, arg0, out, element_type, he_seal_backend,
                              pool);
}

void ngraph::he::scalar_multiply(std::shared_ptr<ngraph::he::HEPlaintext>& arg0,
                                 std::shared_ptr<ngraph::he::HEPlaintext>& arg1,
                                 std::shared_ptr<ngraph::he::HEPlaintext>& out,
                                 const element::Type& element_type,
                                 const HESealBackend* he_seal_backend,
                                 const seal::MemoryPoolHandle& pool) {
  NGRAPH_CHECK(element_type == element::f32);

  std::vector<float> arg0_vals = arg0->get_values();
  std::vector<float> arg1_vals = arg1->get_values();
  std::vector<float> out_vals(arg0->num_values());

  NGRAPH_CHECK(arg0_vals.size() > 0, "Multiplying plaintext arg0 has 0 values");
  NGRAPH_CHECK(arg1_vals.size() > 0, "Multiplying plaintext arg1 has 0 values");

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
