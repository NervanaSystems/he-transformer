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

#include <iostream>

namespace ngraph {
namespace he {
class HECiphertext {
 public:
  HECiphertext(){};
  virtual ~HECiphertext(){};

  virtual void save(std::ostream& stream) const = 0;

  virtual size_t size() const = 0;

  bool is_zero() const { return m_is_zero; }
  void set_zero(bool toggle) { m_is_zero = toggle; }

  bool complex_packing() const { return m_complex_packing; }
  void set_complex_packing(bool toggle) { m_complex_packing = toggle; }

 protected:
  bool m_complex_packing;
  bool m_is_zero;
};

}  // namespace he
}  // namespace ngraph
