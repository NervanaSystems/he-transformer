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
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either minimumress or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <memory>
#include <vector>

#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/minimum_seal.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {

void scalar_minimum_seal(const HEPlaintext& arg0, const HEPlaintext& arg1,
                         HEPlaintext& out) {
  std::vector<double> out_vals(arg0.size());
  std::transform(arg0.begin(), arg0.end(), arg1.begin(), out_vals.begin(),
                 std::minus<double>());
  out = HEPlaintext(std::vector<double>{out_vals});
}

void scalar_minimum_seal(const HEType& arg0, const HEType& arg1, HEType& out,
                         const HESealBackend& he_seal_backend) {
  HEPlaintext arg0_plain, arg1_plain;
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
  NGRAPH_CHECK(arg0.size() == arg1.size(), "arg0.size() = ", arg0.size(),
               " does not match arg1.size()", arg1.size());
  NGRAPH_CHECK(arg0.size() == out.size(), "arg0.size() = ", arg0.size(),
               " does not match out.size()", out.size());
  for (size_t i = 0; i < count; ++i) {
    scalar_minimum_seal(arg0[i], arg1[i], out[i], he_seal_backend);
  }
}

}  // namespace ngraph
}  // namespace ngraph