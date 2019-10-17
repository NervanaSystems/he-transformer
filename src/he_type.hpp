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

#include "he_plaintext.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph {
namespace he {
/// \brief Class representing an HE datatype, either a plaintext or a ciphertext
class HEType {
 public:
  /// \brief Constructs an empty HEType object
  HEType() = default;

  HEType(const HEPlaintext& plain) : m_is_plain(true), m_plain(plain) {}

  HEType(const SealCiphertextWrapper& cipher)
      : m_is_plain(false), m_cipher(cipher) {}

  inline bool is_plaintext() { return m_is_plain; }
  inline bool is_ciphertext() { return !is_plaintext(); }

  const HEPlaintext& get_plaintext() { return m_plain; }

  void set_plaintext(const HEPlaintext& plain) {
    m_plain = plain;
    m_is_plain = true;
    m_cipher.ciphertext().release();
  }

  const SealCiphertextWrapper& get_ciphertext() { return m_cipher; }

  void set_ciphertext(const SealCiphertextWrapper& cipher) {
    m_cipher = cipher;
    m_is_plain = false;
    m_plain.clear();
  }

 private:
  bool m_is_plain;
  HEPlaintext m_plain;
  SealCiphertextWrapper m_cipher;
};

}  // namespace he
}  // namespace ngraph