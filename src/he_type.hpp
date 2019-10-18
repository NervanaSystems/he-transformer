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
  HEType() = delete;

  /// \brief Constructs an empty HEType object
  HEType(const bool plaintext_packing, const bool complex_packing,
         const size_t batch_size)
      : m_plaintext_packing(plaintext_packing),
        m_complex_packing(complex_packing),
        m_batch_size(batch_size) {}

  HEType(const HEPlaintext& plain, const bool plaintext_packing,
         const bool complex_packing, const size_t batch_size)
      : HEType(plaintext_packing, complex_packing, batch_size) {
    m_is_plain = true;
    m_plain = plain;
  }

  HEType(const std::shared_ptr<SealCiphertextWrapper>& cipher,
         const bool plaintext_packing, const bool complex_packing,
         const size_t batch_size)
      : HEType(plaintext_packing, complex_packing, batch_size) {
    m_is_plain = false;
    m_cipher = cipher;
  }

  bool is_plaintext() const { return m_is_plain; }
  bool is_ciphertext() const { return !is_plaintext(); }

  const HEPlaintext& get_plaintext() const { return m_plain; }

  const bool& complex_packing() const { return m_complex_packing; }
  bool& complex_packing() { return m_complex_packing; }

  const bool& plaintext_packing() const { return m_plaintext_packing; }
  bool& plaintext_packing() { return m_plaintext_packing; }

  const size_t& batch_size() const { return m_batch_size; }
  size_t& batch_size() { return m_batch_size; }

  // TODO: &&
  void set_plaintext(const HEPlaintext& plain) {
    m_plain = plain;
    m_is_plain = true;
    NGRAPH_INFO << "Releasing ciphertext";
    if (m_cipher != nullptr) {
      m_cipher->ciphertext().release();
    }
  }

  const std::shared_ptr<SealCiphertextWrapper>& get_ciphertext() const {
    return m_cipher;
  }
  std::shared_ptr<SealCiphertextWrapper> get_ciphertext() { return m_cipher; }

  // TODO: &&
  void set_ciphertext(const std::shared_ptr<SealCiphertextWrapper>& cipher) {
    m_cipher = cipher;
    m_is_plain = false;
    m_plain.clear();
  }

 private:
  bool m_is_plain;
  bool m_plaintext_packing;
  bool m_complex_packing;
  size_t m_batch_size;
  HEPlaintext m_plain;
  std::shared_ptr<SealCiphertextWrapper> m_cipher;
};

}  // namespace he
}  // namespace ngraph