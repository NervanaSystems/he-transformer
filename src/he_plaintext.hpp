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

#include <mutex>
#include <vector>

#include "ngraph/assertion.hpp"

namespace ngraph {
namespace runtime {
namespace he {
class HEPlaintext {
 public:
  HEPlaintext(){};
  HEPlaintext(const std::vector<float> values)
      : m_values(values), m_is_encoded(false), m_is_complex(false){};
  virtual ~HEPlaintext(){};

  void set_values(const std::vector<float>& values) { m_values = values; }
  const std::vector<float>& get_values() const { return m_values; }

  bool is_single_value() { return num_values() == 1; }
  size_t num_values() const { return m_values.size(); }

  bool is_encoded() const { return m_is_encoded; }
  void set_encoded(bool encoded) { m_is_encoded = encoded; }

  bool is_complex() const { return m_is_complex; }
  void set_complex(bool toggle) { m_is_complex = toggle; }

  std::mutex& get_encode_mutex() { return m_encode_mutex; }

 protected:
  std::vector<float> m_values;

  // TODO: use atomic bool?
  bool m_is_encoded;
  bool m_is_complex;

  std::mutex m_encode_mutex;
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
