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
  HEPlaintext(const std::vector<float>& values) : m_values(values) {}
  HEPlaintext() { m_values.reserve(1); }
  HEPlaintext(const float value) : m_values{std::vector<float>{value}} {}

  std::vector<float>& values() { return m_values; }
  const std::vector<float>& values() const { return m_values; }

  bool is_single_value() const { return num_values() == 1; }
  size_t num_values() const { return m_values.size(); }

 private:
  std::vector<float> m_values;
};
}  // namespace he
}  // namespace ngraph
