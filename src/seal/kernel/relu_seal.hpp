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

#pragma once

#include <memory>
#include <vector>

#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {
inline void scalar_relu_seal(const HEPlaintext& arg, HEPlaintext& out) {
  const std::vector<double>& arg_vals = arg.values();
  std::vector<double> out_vals(arg.num_values());

  auto relu = [](double f) { return f > 0 ? f : 0.f; };
  std::transform(arg_vals.begin(), arg_vals.end(), out_vals.begin(), relu);
  out.set_values(out_vals);
}

inline void relu_seal(const std::vector<HEPlaintext>& arg,
                      std::vector<HEPlaintext>& out, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    scalar_relu_seal(arg[i], out[i]);
  }
}

inline void scalar_relu_seal_known_value(
    const SealCiphertextWrapper& arg,
    std::shared_ptr<SealCiphertextWrapper>& out) {
  auto relu = [](double f) { return f > 0 ? f : 0.f; };
  NGRAPH_CHECK(arg.known_value());
  out->known_value() = true;
  out->value() = relu(arg.value());
}

inline void scalar_relu_seal(const SealCiphertextWrapper& arg,
                             std::shared_ptr<SealCiphertextWrapper>& out,
                             const HESealBackend& he_seal_backend) {
  auto relu = [](double f) { return f > 0 ? f : 0.f; };

  if (arg.known_value()) {
    scalar_relu_seal_known_value(arg, out);
  } else {
    HEPlaintext plain;
    he_seal_backend.decrypt(plain, arg);
    const std::vector<double>& arg_vals = plain.values();
    std::vector<double> out_vals(plain.num_values());

    std::transform(arg_vals.begin(), arg_vals.end(), out_vals.begin(), relu);

    plain.set_values(out_vals);
    ngraph::he::encrypt(
        out, plain, he_seal_backend.get_context()->first_parms_id(),
        ngraph::element::f32, he_seal_backend.get_scale(),
        *he_seal_backend.get_ckks_encoder(), *he_seal_backend.get_encryptor(),
        he_seal_backend.complex_packing());
  }
}

inline void relu_seal(
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out, size_t count,
    const HESealBackend& he_seal_backend) {
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_relu_seal(*arg[i], out[i], he_seal_backend);
  }
}

}  // namespace he
}  // namespace ngraph
