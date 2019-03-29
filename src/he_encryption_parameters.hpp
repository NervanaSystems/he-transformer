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

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace ngraph {
namespace runtime {
namespace he {
class HEEncryptionParameters {
 public:
  HEEncryptionParameters() = delete;
  HEEncryptionParameters(std::string scheme_name,
                         std::uint64_t poly_modulus_degree,
                         std::uint64_t security_level,
                         std::uint64_t evaluation_decomposition_bit_count,
                         std::vector<std::uint64_t> coeff_modulus,
                         std::uint64_t plain_modulus = 0)
      : m_scheme_name(scheme_name),
        m_poly_modulus_degree(poly_modulus_degree),
        m_security_level(security_level),
        m_evaluation_decomposition_bit_count(
            evaluation_decomposition_bit_count),
        m_coeff_modulus(coeff_modulus),
        m_plain_modulus(plain_modulus) {}

  virtual ~HEEncryptionParameters(){};

  virtual void save(std::ostream& stream) const = 0;

  inline const std::string& scheme_name() const { return m_scheme_name; }

  inline std::uint64_t poly_modulus_degree() { return m_poly_modulus_degree; }

  inline std::uint64_t security_level() { return m_security_level; }

  inline std::uint64_t evaluation_decomposition_bit_count() {
    return m_evaluation_decomposition_bit_count;
  }
  inline const std::vector<std::uint64_t>& coeff_modulus() {
    return m_coeff_modulus;
  }

 protected:
  std::string m_scheme_name;
  std::uint64_t m_poly_modulus_degree;
  std::uint64_t m_security_level;
  std::uint64_t m_evaluation_decomposition_bit_count;
  std::vector<std::uint64_t> m_coeff_modulus;
  std::uint64_t m_plain_modulus;
};  // namespace he
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
