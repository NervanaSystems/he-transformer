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

#include "ngraph/assertion.hpp"
#include "seal/seal.h"

namespace ngraph {
namespace he {
class HEPlaintext {
 public:
  HEPlaintext() = default;

  HEPlaintext(const std::vector<double>& values) : m_values(values) {
    if (values.size() > 0) {
      m_first_val = values[0];
    }
  }
  HEPlaintext(const double value)
      : m_first_val(value), m_values{std::vector<double>{value}} {}

  const std::vector<double>& values() const { return m_values; }

  double first_value() const { return m_first_val; }

  void set_value(const double value) {
    m_first_val = value;
    m_values = std::vector<double>{value};
  }

  void set_values(const std::vector<double>& values) {
    m_values = values;
    if (values.size() > 0) {
      m_first_val = m_values[0];
    }
  }

  bool is_single_value() const { return num_values() == 1; }
  size_t num_values() const { return m_values.size(); }

  static constexpr size_t type_byte_size = sizeof(double);

 private:
  double m_first_val;
  std::vector<double> m_values;
};
}  // namespace he
}  // namespace ngraph
