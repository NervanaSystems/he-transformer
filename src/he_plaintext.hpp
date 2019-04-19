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

#include <vector>

#include "ngraph/assertion.hpp"

namespace ngraph {
namespace runtime {
namespace he {
class HEPlaintext {
 public:
  HEPlaintext(){};
  HEPlaintext(const std::vector<float> values) : m_values(values){};
  virtual ~HEPlaintext(){};

  void set_values(const std::vector<float>& values) { m_values = values; }
  std::vector<float>& get_values() { return m_values; }

  bool is_single_value() {
    NGRAPH_ASSERT(m_values.size() != 0) << "Plaintext not initialized";
    return m_values.size() == 1;
  }

  size_t num_values() { return m_values.size(); }

 protected:
  std::vector<float> m_values;
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
