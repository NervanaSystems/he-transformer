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

namespace ngraph {
namespace he {
inline void scalar_bounded_relu_seal(const HEPlaintext& arg, HEPlaintext& out,
                                     float alpha) {
  const std::vector<float>& arg_vals = arg.values();
  std::vector<float> out_vals(arg.num_values());

  auto bounded_relu = [alpha](float f) {
    return f > alpha ? alpha : (f > 0) ? f : 0.f;
  };

  std::transform(arg_vals.begin(), arg_vals.end(), out_vals.begin(),
                 bounded_relu);
  out.values() = out_vals;
}

inline void bounded_relu_seal(const std::vector<HEPlaintext>& arg,
                              std::vector<HEPlaintext>& out, size_t count,
                              float alpha) {
  NGRAPH_CHECK(count <= arg.size(), "arg too small in BoundedRelu");
  NGRAPH_CHECK(count <= out.size(), "out too small in BoundedRelu");
  for (size_t i = 0; i < count; ++i) {
    scalar_bounded_relu_seal(arg[i], out[i], alpha);
  }
}

inline void scalar_bounded_relu_seal(
    const SealCiphertextWrapper& arg,
    std::shared_ptr<SealCiphertextWrapper>& out, float alpha,
    const HESealBackend& he_seal_backend) {
  HEPlaintext plain;
  he_seal_backend.decrypt(plain, arg);
  const std::vector<float>& arg_vals = plain.values();
  std::vector<float> out_vals(plain.num_values());

  std::cout << ": " << arg_vals[0] << ", " << arg_vals[1] << std::endl;
  NGRAPH_INFO << "relu6 out complex packed? " << out->complex_packing();

  auto bounded_relu = [alpha](float f) {
    return f > alpha ? alpha : (f > 0) ? f : 0.f;
  };

  std::transform(arg_vals.begin(), arg_vals.end(), out_vals.begin(),
                 bounded_relu);

  plain.values() = out_vals;

  // TODO: remove
  // NGRAPH_INFO << "Relu6 vals";
  // for (const auto& elem : out_vals) {

  //}

  he_seal_backend.encrypt(out, plain, he_seal_backend.complex_packing());
}

inline void bounded_relu_seal(
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out, size_t count,
    float alpha, const HESealBackend& he_seal_backend) {
  //#pragma omp parallel for
  NGRAPH_INFO << "Relu6 on " << count << " elements";
  for (size_t i = 0; i < count; ++i) {  // TODO: replace with count
    std::cout << "Relu6 index " << i;
    scalar_bounded_relu_seal(*arg[i], out[i], alpha, he_seal_backend);
  }
  // throw ngraph_error("Done with relu6");
}

}  // namespace he
}  // namespace ngraph
