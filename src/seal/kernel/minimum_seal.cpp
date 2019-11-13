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

#include "seal/kernel/minimum_seal.hpp"

#include <vector>

#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"

namespace ngraph::he {

void scalar_minimum_seal(const HEPlaintext& arg0, const HEPlaintext& arg1,
                         HEPlaintext& out) {
  HEPlaintext out_vals;
  if (arg0.size() == 1) {
    out_vals.resize(arg1.size());
    std::transform(arg1.begin(), arg1.end(), out_vals.begin(),
                   [&](auto x) { return std::min(x, arg0[0]); });
  } else if (arg1.size() == 1) {
    out_vals.resize(arg0.size());
    std::transform(arg0.begin(), arg0.end(), out_vals.begin(),
                   [&](auto x) { return std::min(x, arg1[0]); });
  } else {
    size_t min_size = std::min(arg0.size(), arg1.size());
    out_vals.resize(min_size);
    for (size_t i = 0; i < min_size; ++i) {
      out_vals[i] = std::min(arg0[i], arg1[i]);
    }
  }
  out = std::move(out_vals);
}

void scalar_minimum_seal(const HEType& arg0, const HEType& arg1, HEType& out,
                         const HESealBackend& he_seal_backend) {
  HEPlaintext arg0_plain;
  HEPlaintext arg1_plain;

  if (arg0.is_ciphertext()) {
    he_seal_backend.decrypt(arg0_plain, *arg0.get_ciphertext(),
                            arg0.complex_packing());
  } else {
    arg0_plain = arg0.get_plaintext();
  }
  if (arg1.is_ciphertext()) {
    he_seal_backend.decrypt(arg1_plain, *arg1.get_ciphertext(),
                            arg1.complex_packing());
  } else {
    arg1_plain = arg1.get_plaintext();
  }

  HEPlaintext out_plain;
  scalar_minimum_seal(arg0_plain, arg1_plain, out_plain);

  if (out.is_ciphertext()) {
    he_seal_backend.encrypt(out.get_ciphertext(), out_plain, element::f32,
                            out.complex_packing());
  } else {
    out.set_plaintext(out_plain);
  }
}

void minimum_seal(const std::vector<HEType>& arg0,
                  const std::vector<HEType>& arg1, std::vector<HEType>& out,
                  size_t count, HESealBackend& he_seal_backend) {
  for (size_t i = 0; i < count; ++i) {
    scalar_minimum_seal(arg0[i], arg1[i], out[i], he_seal_backend);
  }
}

}  // namespace ngraph::he
