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

#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph {
namespace he {
/// \brief Class representing an HE datatype, either a plaintext or a ciphertext
class HEType {
 public:
  HEType(const HEPlaintext& plain) : m_plain(plain), m_is_plain(true) {}

  HEType(const SealCiphertextWrapper& cipher)
      : m_cipher(cipher), m_is_plain(false) {}

  inline bool is_plaintext() { return m_is_plaintext; }
  inline bool is_ciphertext() { return !is_plain(); }

  const HEPlaintext& get_plaintext() { return m_plain; }

  void set_plaintext(const HEPlaintext& plain) {
    m_plain = plain;
    m_is_plaintext = true;
    m_cipher.release();
  }

  const SealCiphertextWrapper& get_ciphertext() { return m_cipher; }

  void set_ciphertext(const SealCiphertextWrapper& cipher) {
    m_cipher = cipher;
    m_is_plaintext = false;
    m_plain.clear();
  }

 private:
  bool m_is_plaintext;
  HEPlaintext m_plain;
  SealCiphertextWrapper m_cipher;
}

}  // namespace he
}  // namespace ngraph