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

namespace ngraph {
namespace runtime {
namespace he {

/// @brief Class holding a 32-bit fixed-point number
/// Note: I + F must equal 32
template <size_t I = 8, size_t F = 24>
class FixedPoint {
 public:
  FixedPoint(float f) : m_float(f) {
    m_uint = size_t(f * (1 << F)) + size_t(1 << (I + F - 1));
  }

  FixedPoint(uint32_t num) : m_uint(num) {
    float numerator = num - (1 << (I + F - 1));
    float denom = 1 << F;
    m_float = numerator / denom;
  }

  uint32_t as_int() { return m_uint; }
  float as_float() { return m_float; }

 private:
  uint32_t m_uint = 0;
  float m_float = 0;
};

}  // namespace he
}  // namespace runtime
}  // namespace ngraph