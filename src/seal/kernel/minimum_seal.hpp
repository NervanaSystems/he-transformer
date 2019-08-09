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
inline void scalar_minimum_seal(const HEPlaintext& arg0,
                                const HEPlaintext& arg1, HEPlaintext& out) {
  const std::vector<double>& arg0_vals = arg0.values();
  const std::vector<double>& arg1_vals = arg1.values();
  std::vector<double> out_vals(arg0.num_values());

  NGRAPH_CHECK(arg0.num_values() == arg1.num_values(),
               "arg0.num_values() = ", arg0.num_values(),
               " does not match arg1.num_values()", arg1.num_values());

  for (size_t i = 0; i < arg0.num_values(); ++i) {
    out_vals[i] = arg0_vals[i] < arg1_vals[i] ? arg0_vals[i] : arg1_vals[i];
  }

  out.set_values(out_vals);
}

inline void minimum_seal(const std::vector<HEPlaintext>& arg0,
                         const std::vector<HEPlaintext>& arg1,
                         std::vector<HEPlaintext>& out, size_t count) {
  NGRAPH_CHECK(arg0.size() == arg1.size(), "arg0.size() = ", arg0.size(),
               " does not match arg1.size()", arg1.size());
  NGRAPH_CHECK(arg0.size() == out.size(), "arg0.size() = ", arg0.size(),
               " does not match out.size()", out.size());
  for (size_t i = 0; i < count; ++i) {
    scalar_minimum_seal(arg0[i], arg1[i], out[i]);
  }
}

}  // namespace he
}  // namespace ngraph
