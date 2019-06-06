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

#include "seal/seal.h"

namespace ngraph {
namespace he {
class SealCiphertextWrapper {
 public:
  SealCiphertextWrapper() : m_complex_packing(false), m_is_zero(false) {}

  SealCiphertextWrapper(const seal::Ciphertext& cipher,
                        bool complex_packing = false, bool is_zero = false)
      : m_ciphertext(cipher),
        m_complex_packing(complex_packing),
        m_is_zero(is_zero) {}

  seal::Ciphertext& ciphertext() { return m_ciphertext; }
  const seal::Ciphertext& ciphertext() const { return m_ciphertext; }

  void save(std::ostream& stream) const { m_ciphertext.save(stream); }

  size_t size() const { return m_ciphertext.size(); }

  bool is_zero() const { return m_is_zero; }
  bool& is_zero() { return m_is_zero; }

  double& scale() { return m_ciphertext.scale(); }
  const double scale() const { return m_ciphertext.scale(); }

  bool complex_packing() const { return m_complex_packing; }
  bool& complex_packing() { return m_complex_packing; }

 private:
  bool m_complex_packing;
  bool m_is_zero;
  seal::Ciphertext m_ciphertext;
};

}  // namespace he
}  // namespace ngraph
