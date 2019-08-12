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

  NGRAPH_CHECK(context_data.parms().scheme() == seal::scheme_type::CKKS,
               "Only CKKS scheme supported");

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

static inline double type_to_double(const void* src,
                                    const element::Type& element_type) {
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
  switch (element_type.get_type_enum()) {
    case element::Type_t::f32:
      return static_cast<double>(*static_cast<const float*>(src));
      break;
    case element::Type_t::f64:
      return static_cast<double>(*static_cast<const double*>(src));
      break;
    case element::Type_t::i8:
    case element::Type_t::i16:
    case element::Type_t::i32:
    case element::Type_t::i64:
      // TODO: reinterpret cast
      return static_cast<double>(*static_cast<const int64_t*>(src));
    case element::Type_t::u8:
    case element::Type_t::u16:
    case element::Type_t::u32:
    case element::Type_t::u64:
    case element::Type_t::dynamic:
    case element::Type_t::undefined:
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::boolean:
      NGRAPH_CHECK(false, "Unsupported element type", element_type);
      break;
  }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
}

static inline std::vector<double> type_vec_to_double_vec(
    const void* src, const element::Type& element_type, size_t n) {
  std::vector<double> ret(n);
  char* src_with_offset = static_cast<char*>(const_cast<void*>(src));
  for (size_t i = 0; i < n; ++i) {
    ret[i] = ngraph::he::type_to_double(src_with_offset, element_type);
    ++src_with_offset;
  }
  return ret;
}

}  // namespace he
}  // namespace ngraph