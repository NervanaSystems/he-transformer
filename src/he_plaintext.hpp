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

#include <ostream>
#include <vector>
#include "ngraph/type/element_type.hpp"

namespace ngraph {
namespace he {
/// \brief Class representing a plaintext value
class HEPlaintext : public std::vector<double> {
 public:
  HEPlaintext() = default;
  HEPlaintext(const std::initializer_list<double>& values)
      : std::vector<double>(values) {}

  HEPlaintext(const HEPlaintext& plain) : std::vector<double>(plain) {}

  HEPlaintext(const std::vector<double>& values)
      : std::vector<double>(values) {}

  HEPlaintext(size_t n, double initial_value = 0)
      : std::vector<double>(n, initial_value) {}

  template <class InputIterator>
  HEPlaintext(InputIterator first, InputIterator last)
      : std::vector<double>(first, last) {}

  HEPlaintext& operator=(const HEPlaintext& v) {
    static_cast<std::vector<double>*>(this)->operator=(v);
    return *this;
  }

  HEPlaintext& operator=(HEPlaintext&& v) {
    static_cast<std::vector<double>*>(this)->operator=(v);
    return *this;
  }

  /// \brief Writes the plaintext to the target as a vector of type
  void write(void* target, const element::Type& element_type);
};

std::ostream& operator<<(std::ostream& os, const HEPlaintext& plain);
}  // namespace he
}  // namespace ngraph
