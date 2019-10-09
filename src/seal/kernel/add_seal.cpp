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

#include "seal/kernel/add_seal.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {
void scalar_add_seal(SealCiphertextWrapper& arg0, SealCiphertextWrapper& arg1,
                     std::shared_ptr<SealCiphertextWrapper>& out,
                     const ngraph::element::Type& element_type,
                     HESealBackend& he_seal_backend,
                     const seal::MemoryPoolHandle& pool) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
  if (arg0.known_value() && arg1.known_value()) {
    out->known_value() = true;
    out->value() = arg0.value() + arg1.value();
  } else if (arg0.known_value()) {
    HEPlaintext p(arg0.value());
    scalar_add_seal(p, arg1, out, element_type, he_seal_backend);
    out->known_value() = false;
  } else if (arg1.known_value()) {
    HEPlaintext p(arg1.value());
    scalar_add_seal(p, arg0, out, element_type, he_seal_backend);
    out->known_value() = false;
  } else {
    NGRAPH_CHECK(arg0.complex_packing() == arg1.complex_packing(),
                 "arg0.complex_packing() (", arg0.complex_packing(),
                 ") does not match arg1.complex_packing() (",
                 arg1.complex_packing(), ")");
    NGRAPH_CHECK(arg0.complex_packing() == he_seal_backend.complex_packing(),
                 "Add arg0 is not he_seal_backend.complex_packing()");
    NGRAPH_CHECK(arg1.complex_packing() == he_seal_backend.complex_packing(),
                 "Add arg1 is not he_seal_backend.complex_packing()");

    match_modulus_and_scale_inplace(arg0, arg1, he_seal_backend, pool);
    he_seal_backend.get_evaluator()->add(arg0.ciphertext(), arg1.ciphertext(),
                                         out->ciphertext());

    out->known_value() = false;
  }
  out->complex_packing() = he_seal_backend.complex_packing();
}

void scalar_add_seal(SealCiphertextWrapper& arg0, const HEPlaintext& arg1,
                     std::shared_ptr<SealCiphertextWrapper>& out,
                     const ngraph::element::Type& element_type,
                     HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
  if (arg0.known_value()) {
    NGRAPH_CHECK(arg1.is_single_value(), "arg1 is not single value");
    out->known_value() = true;
    out->value() = arg0.value() + arg1.first_value();
    out->complex_packing() = arg0.complex_packing();
    return;
  }
  // TODO: handle case where arg1 = {0, 0, 0, 0, ...}
  bool add_zero = arg1.is_single_value() && (arg1.first_value() == 0.0);

  if (add_zero) {
    SealCiphertextWrapper tmp(arg0);
    NGRAPH_CHECK(tmp.complex_packing() == arg0.complex_packing());
    out = std::make_shared<SealCiphertextWrapper>(tmp);
    out->complex_packing() = tmp.complex_packing();

  } else {
    bool complex_packing = arg0.complex_packing();
    // TODO: optimize for adding single complex number
    if (arg1.is_single_value() && !complex_packing) {
      double value = arg1.first_value();
      add_plain(arg0.ciphertext(), value, out->ciphertext(), he_seal_backend);
    } else {
      auto p = SealPlaintextWrapper(complex_packing);
      encode(p, arg1, *he_seal_backend.get_ckks_encoder(),
             arg0.ciphertext().parms_id(), element_type,
             arg0.ciphertext().scale(), complex_packing);
      size_t chain_ind0 = he_seal_backend.get_chain_index(arg0);
      size_t chain_ind1 = he_seal_backend.get_chain_index(p.plaintext());
      NGRAPH_CHECK(chain_ind0 == chain_ind1, "Chain inds ", chain_ind0, ",  ",
                   chain_ind1, " don't match");

      he_seal_backend.get_evaluator()->add_plain(
          arg0.ciphertext(), p.plaintext(), out->ciphertext());
    }
    out->complex_packing() = arg0.complex_packing();
  }
  out->known_value() = false;
}

void scalar_add_seal(const HEPlaintext& arg0, const HEPlaintext& arg1,
                     HEPlaintext& out,
                     const ngraph::element::Type& element_type,
                     HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);

  const std::vector<double>& arg0_vals = arg0.values();
  const std::vector<double>& arg1_vals = arg1.values();
  std::vector<double> out_vals;

  if (arg0_vals.size() == 1) {
    std::transform(
        arg1_vals.begin(), arg1_vals.end(), std::back_inserter(out_vals),
        std::bind(std::plus<double>(), std::placeholders::_1, arg0_vals[0]));
  } else if (arg1_vals.size() == 1) {
    std::transform(
        arg0_vals.begin(), arg0_vals.end(), std::back_inserter(out_vals),
        std::bind(std::plus<double>(), std::placeholders::_1, arg1_vals[0]));
  } else {
    NGRAPH_CHECK(arg0.num_values() == arg1.num_values(), "arg0 num values ",
                 arg0.num_values(), " != arg1 num values ", arg1.num_values(),
                 " in plain-plain add");
    std::transform(arg0_vals.begin(), arg0_vals.end(), arg1_vals.begin(),
                   std::back_inserter(out_vals), std::plus<double>());
  }
  out.set_values(out_vals);
}

}  // namespace he
}  // namespace ngraph