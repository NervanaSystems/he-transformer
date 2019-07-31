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

#include <complex>
#include <string>
#include <unordered_set>
#include <vector>

#include "ngraph/check.hpp"
#include "ngraph/except.hpp"
#include "ngraph/util.hpp"
#include "seal/seal.h"

namespace ngraph {
namespace he {
static inline void print_seal_context(const seal::SEALContext& context) {
  auto& context_data = *context.key_context_data();

  assert(context_data.parms().scheme() == seal::scheme_type::CKKS);

  std::cout << "/" << std::endl;
  std::cout << "| Encryption parameters :" << std::endl;
  std::cout << "|   scheme: CKKS" << std::endl;
  std::cout << "|   poly_modulus_degree: "
            << context_data.parms().poly_modulus_degree() << std::endl;
  std::cout << "|   coeff_modulus size: ";
  std::cout << context_data.total_coeff_modulus_bit_count() << " (";
  auto coeff_modulus = context_data.parms().coeff_modulus();
  std::size_t coeff_mod_count = coeff_modulus.size();
  for (std::size_t i = 0; i < coeff_mod_count - 1; i++) {
    std::cout << coeff_modulus[i].bit_count() << " + ";
  }
  std::cout << coeff_modulus.back().bit_count();
  std::cout << ") bits" << std::endl;
  std::cout << "\\" << std::endl;
}

// Packs elements of input into real values
// (a+bi, c+di) => (a,b,c,d)
static inline void complex_vec_to_real_vec(
    std::vector<double>& output,
    const std::vector<std::complex<double>>& input) {
  NGRAPH_CHECK(output.size() == 0);
  output.reserve(input.size() * 2);
  for (const std::complex<double>& value : input) {
    output.emplace_back(value.real());
    output.emplace_back(value.imag());
  }
}

// Packs elements of input into complex values
// (a,b,c,d) => (a+bi, c+di)
// (a,b,c) => (a+bi, c+0i)
static inline void real_vec_to_complex_vec(
    std::vector<std::complex<double>>& output,
    const std::vector<double>& input) {
  NGRAPH_CHECK(output.size() == 0);
  output.reserve(input.size() / 2);
  std::vector<double> complex_parts(2, 0);
  for (size_t i = 0; i < input.size(); ++i) {
    complex_parts[i % 2] = input[i];

    if (i % 2 == 1 || i == input.size() - 1) {
      output.emplace_back(
          std::complex<double>(complex_parts[0], complex_parts[1]));
      complex_parts = {0, 0};
    }
  }
}

static inline bool flag_to_bool(const char* flag, bool default_value = false) {
  if (flag == nullptr) {
    return default_value;
  }
  static std::unordered_set<std::string> on_map{"1", "y", "yes"};
  static std::unordered_set<std::string> off_map{"0", "n", "no"};
  std::string flag_str = ngraph::to_lower(std::string(flag));

  if (on_map.find(flag_str) != on_map.end()) {
    return true;
  } else if (off_map.find(flag_str) != off_map.end()) {
    return true;
  } else {
    throw ngraph_error("Unknown flag value " + std::string(flag));
  }
}
}  // namespace he
}  // namespace ngraph