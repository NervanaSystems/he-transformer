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
  HEPlaintext(const std::vector<float> values = std::vector<float>{},
              bool complex_packing = false)
      : m_values(values), m_complex_packing(complex_packing){};
  HEPlaintext(const float value, bool complex_packing = false)
      : m_values{std::vector<float>{value}},
        m_complex_packing(complex_packing){};

  HEPlaintext(bool complex_packing)
      : m_values(std::vector<float>{}), m_complex_packing(complex_packing){};
  virtual ~HEPlaintext(){};

  void set_values(const std::vector<float>& values) { m_values = values; }
  const std::vector<float>& get_values() const { return m_values; }

  bool is_single_value() const { return num_values() == 1; }
  size_t num_values() const { return m_values.size(); }

  bool complex_packing() const { return m_complex_packing; }
  bool& complex_packing() { return m_complex_packing; }

 protected:
  std::vector<float> m_values;
  // TODO: move to plain tensor
  bool m_complex_packing;
};
}  // namespace he
}  // namespace ngraph
