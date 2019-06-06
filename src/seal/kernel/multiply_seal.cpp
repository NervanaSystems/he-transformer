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
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/negate_seal.hpp"
#include "seal/seal_util.hpp"

void ngraph::he::scalar_multiply_seal(
    ngraph::he::SealCiphertextWrapper& arg0,
    ngraph::he::SealCiphertextWrapper& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend& he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  if (arg0.is_zero() || arg1.is_zero()) {
    out->is_zero() = true;
  } else {
    NGRAPH_CHECK(arg0.complex_packing() == false,
                 "cannot multiply ciphertexts in complex form");
    NGRAPH_CHECK(arg1.complex_packing() == false,
                 "cannot multiply ciphertexts in complex form");
    out->is_zero() = false;

    match_modulus_and_scale_inplace(arg0, arg1, he_seal_backend, pool);
    size_t chain_ind0 = get_chain_index(arg0, he_seal_backend);
    size_t chain_ind1 = get_chain_index(arg1, he_seal_backend);

    if (chain_ind0 == 0 || chain_ind1 == 0) {
      NGRAPH_INFO << "Multiplicative depth limit reached";
      exit(1);
    }

    if (&arg0 == &arg1) {
      he_seal_backend.get_evaluator()->square(arg0.ciphertext(),
                                              out->ciphertext(), pool);
    } else {
      he_seal_backend.get_evaluator()->multiply(
          arg0.ciphertext(), arg1.ciphertext(), out->ciphertext(), pool);
    }

    he_seal_backend.get_evaluator()->relinearize_inplace(
        out->ciphertext(), *(he_seal_backend.get_relin_keys()), pool);

    // TODO: lazy rescaling if before dot
    he_seal_backend.get_evaluator()->rescale_to_next_inplace(out->ciphertext(),
                                                             pool);
  }
}

void ngraph::he::scalar_multiply_seal(
    ngraph::he::SealCiphertextWrapper& arg0,
    const ngraph::he::HEPlaintext& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend& he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  NGRAPH_CHECK(element_type == element::f32, "Element type ", element_type,
               " is not float");
  if (arg0.is_zero()) {
    out->is_zero() = true;
    return;
  }
  // We can't do the scalar +/-1 optimizations, unless all the weights
  // are +/-1 in this layer, since we expect the scale of the ciphertext to
  // square. For instance, if we are computing c1*p(1) + c2 *p(2), the latter
  // sum will have larger scale than the former

  const auto& values = arg1.values();
  // TODO: check multiplying by small numbers behavior more thoroughly
  if (std::all_of(values.begin(), values.end(),
                  [](float f) { return std::abs(f) < 1e-5f; })) {
    out->is_zero() = true;
  } else {
    out->is_zero() = false;
    if (arg1.is_single_value()) {
      float value = arg1.values()[0];
      double double_val = double(value);
      multiply_plain(arg0.ciphertext(), double_val, out->ciphertext(),
                     he_seal_backend, pool);
    } else {
      // Never complex-pack for multiplication
      auto p = SealPlaintextWrapper(false);
      he_seal_backend.encode(p, arg1, arg0.ciphertext().parms_id(),
                             arg0.ciphertext().scale());

      size_t chain_ind0 = get_chain_index(arg0, he_seal_backend);
      size_t chain_ind1 = get_chain_index(p.plaintext(), he_seal_backend);

      NGRAPH_CHECK(chain_ind0 == chain_ind1, "Chain_ind0 ", chain_ind0,
                   " != chain_ind1 ", chain_ind1);
      NGRAPH_CHECK(chain_ind0 > 0, "Multiplicative depth exceeded for arg0");
      NGRAPH_CHECK(chain_ind1 > 0, "Multiplicative depth exceeded for arg1");

      try {
        he_seal_backend.get_evaluator()->multiply_plain(
            arg0.ciphertext(), p.plaintext(), out->ciphertext(), pool);
      } catch (const std::exception& e) {
        NGRAPH_INFO << "Error multiplying plain " << e.what();
        NGRAPH_INFO << "arg1->values().size() " << arg1.num_values();
        for (const auto& elem : arg1.values()) {
          NGRAPH_INFO << elem;
        }
      }
    }
    out->complex_packing() = arg0.complex_packing();
  }
}

void ngraph::he::scalar_multiply_seal(
    const ngraph::he::HEPlaintext& arg0,
    ngraph::he::SealCiphertextWrapper& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend& he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  ngraph::he::scalar_multiply_seal(arg1, arg0, out, element_type,
                                   he_seal_backend, pool);
}

void ngraph::he::scalar_multiply_seal(const ngraph::he::HEPlaintext& arg0,
                                      const ngraph::he::HEPlaintext& arg1,
                                      ngraph::he::HEPlaintext& out,
                                      const element::Type& element_type,
                                      const HESealBackend& he_seal_backend,
                                      const seal::MemoryPoolHandle& pool) {
  NGRAPH_CHECK(element_type == element::f32);

  std::vector<float> arg0_vals = arg0.values();
  std::vector<float> arg1_vals = arg1.values();
  std::vector<float> out_vals(arg0.num_values());

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
    NGRAPH_CHECK(arg0.num_values() == arg1.num_values(), "arg0 num values ",
                 arg0.num_values(), " != arg1 num values ", arg1.num_values(),
                 " in plain-plain multiply");

    std::transform(arg0_vals.begin(), arg0_vals.end(), arg1_vals.begin(),
                   out_vals.begin(), std::multiplies<float>());
  }
  out.values() = out_vals;
}
