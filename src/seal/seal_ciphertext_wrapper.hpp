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

#include <cstddef>
#include <memory>

#include "logging/ngraph_he_log.hpp"
#include "ngraph/check.hpp"
#include "protos/message.pb.h"
#include "seal/seal.h"

namespace ngraph {
namespace he {
/// \brief Returns the size in bytes required to serialize a ciphertext
/// \param[in] cipher Ciphertext to measure size of
inline size_t ciphertext_size(const seal::Ciphertext& cipher) {
  return cipher.save_size(seal::compr_mode_type::none);
}

/// \brief Serializes the ciphertext and writes to a destination
/// \param[in] cipher Ciphertext to write
/// \param[out] destination Where to save ciphertext to
/// \returns The size in bytes of the saved ciphertext
inline std::size_t save(const seal::Ciphertext& cipher,
                        std::byte* destination) {
  return cipher.save(destination, ciphertext_size(cipher),
                     seal::compr_mode_type::none);
}

/// \brief Loads a serialized ciphertext
/// \param[out] cipher De-serialized ciphertext
/// \param[in] context Encryption context to verify ciphertext validity against
/// \param[in] src Pointer to data to load from
/// \param[in] size Number of bytes available in the memory location
inline void load(seal::Ciphertext& cipher,
                 std::shared_ptr<seal::SEALContext> context,
                 const std::byte* src, const std::size_t size) {
  cipher.load(context, src, size);
}

/// \brief Class representing a lightweight wrapper around a SEAL ciphertext.
class SealCiphertextWrapper {
 public:
  /// \brief Create an empty ciphertext
  SealCiphertextWrapper() {}

  /// \brief Create ciphertext wrapper from ciphertext
  /// \param[in] cipher Ciphertext to store
  SealCiphertextWrapper(seal::Ciphertext cipher)
      : m_ciphertext(std::move(cipher)) {}

  /// \brief Returns the ciphertext
  inline seal::Ciphertext& ciphertext() { return m_ciphertext; }

  /// \brief Returns the ciphertext
  inline const seal::Ciphertext& ciphertext() const { return m_ciphertext; }

  /// \brief Returns the size of the ciphertext
  inline size_t size() const { return m_ciphertext.size(); }

  /// \brief Returns scale of the ciphertext
  inline double& scale() { return m_ciphertext.scale(); }

  /// \brief Returns scale of the ciphertext
  inline double scale() const { return m_ciphertext.scale(); }

  /// \brief Writes the ciphertext to a protobuf object
  /// \param[out] he_type Protobuf object to write ciphertext to
  inline void save(proto::HEType& he_type) const {
    size_t cipher_size = ciphertext_size(m_ciphertext);
    std::string cipher_str;
    cipher_str.resize(cipher_size);

    size_t save_size = ngraph::he::save(
        m_ciphertext, reinterpret_cast<std::byte*>(cipher_str.data()));

    NGRAPH_CHECK(save_size == cipher_size, "Save size != cipher size");

    he_type.set_ciphertext(std::move(cipher_str));
  }

  /// \brief Loads a ciphertext from a protobuf object
  /// \param[out] dst Destination to load ciphertext to
  /// \param[in] proto_he_type Protobuf object to load object from
  /// \param[in] context SEAL context to validate loaded ciphertext against
  static inline void load(SealCiphertextWrapper& dst,
                          const proto::HEType& proto_he_type,
                          std::shared_ptr<seal::SEALContext> context) {
    NGRAPH_CHECK(!proto_he_type.is_plaintext(),
                 "Cannot load ciphertext from plaintext HEType");

    const std::string& cipher_str = proto_he_type.ciphertext();
    ngraph::he::load(dst.ciphertext(), context,
                     reinterpret_cast<const std::byte*>(cipher_str.data()),
                     cipher_str.size());
  }

 private:
  seal::Ciphertext m_ciphertext;
};

}  // namespace he
}  // namespace ngraph
