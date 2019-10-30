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

#include <memory>
#include <vector>

#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/power_seal.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {

void scalar_power_seal(const HEPlaintext& arg0, const HEPlaintext& arg1,
                       HEPlaintext& out) {
  std::vector<double> out_vals(arg0.size());
  auto power = [](auto x, auto y) -> auto { return std::pow(x, y); };
  std::transform(arg0.begin(), arg0.end(), arg1.begin(), out_vals.begin(),
                 power);
  out = HEPlaintext({out_vals});
}

void scalar_power_seal(HEType& arg0, HEType& arg1, HEType& out,
                       HESealBackend& he_seal_backend) {
  // TODO(fboemer): enable with client?
  // TODO(fboemer): complex packing?

  if (arg0.is_ciphertext() && arg1.is_ciphertext()) {
    NGRAPH_CHECK(arg0.complex_packing() == arg1.complex_packing(),
                 "Complex packing types don't match");

    HEPlaintext plain_arg0, plain_arg1;
    he_seal_backend.decrypt(plain_arg0, *arg0.get_ciphertext(),
                            arg0.complex_packing());
    he_seal_backend.decrypt(plain_arg1, *arg1.get_ciphertext(),
                            arg1.complex_packing());
    scalar_power_seal(plain_arg0, plain_arg1, plain_arg1);

    he_seal_backend.encrypt(out.get_ciphertext(), plain_arg1,
                            ngraph::element::f32, arg0.complex_packing());

  } else if (arg0.is_ciphertext() && arg1.is_plaintext()) {
    HEPlaintext plain_arg0;
    he_seal_backend.decrypt(plain_arg0, *arg0.get_ciphertext(),
                            arg0.complex_packing());
    scalar_power_seal(plain_arg0, arg1.get_plaintext(), plain_arg0);
    he_seal_backend.encrypt(out.get_ciphertext(), plain_arg0,
                            ngraph::element::f32, arg0.complex_packing());

  } else if (arg0.is_plaintext() && arg1.is_ciphertext()) {
    HEPlaintext plain_arg1;
    he_seal_backend.decrypt(plain_arg1, *arg1.get_ciphertext(),
                            arg1.complex_packing());
    scalar_power_seal(arg0.get_plaintext(), plain_arg1, plain_arg1);
    he_seal_backend.encrypt(out.get_ciphertext(), plain_arg1,
                            ngraph::element::f32, arg0.complex_packing());

  } else if (arg0.is_plaintext() && arg1.is_plaintext()) {
    out.set_plaintext(arg0.get_plaintext());
    scalar_power_seal(arg0.get_plaintext(), arg1.get_plaintext(),
                      out.get_plaintext());
  } else {
    NGRAPH_CHECK(false, "Unknown argument types");
  }
}

void power_seal(std::vector<HEType>& arg0, std::vector<HEType>& arg1,
                std::vector<HEType>& out, size_t count,
                const element::Type& element_type,
                HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);

#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_power_seal(arg0[i], arg1[i], out[i], he_seal_backend);
  }
}

}  // namespace he
}  // namespace ngraph
