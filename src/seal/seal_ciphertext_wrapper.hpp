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

/// TODO: update
/// \brief Class representing a lightweight wrapper around a SEAL ciphertext.
/// The wrapper contains two attributes in addition to the SEAL ciphertext.
/// First, whether or not the ciphertext stores values using complex packing
/// Second, whether or not the ciphertext represents a publicly-known value.
/// Typically, a ciphertext represents encrypted data, which is not known unless
/// decryption has been performed. However, two special cases result in a
/// "known-valued" ciphertext. First, multiplying a ciphertext with a plaintext
/// zero results in a "known-valued" ciphertext ith known value 0. Second, the
/// "Pad" operation may pad a known plaintext value to HESealCipherTensor. The
/// padded value itself is public, so the resulting ciphertext will be this
/// known value. This is a design choice which allows HESealCipherTensors to
/// store a vector of SealCiphertextWrappers.
class SealCiphertextWrapper {
 public:
  /// \brief Create an empty unknown-valued ciphertext
  SealCiphertextWrapper() {}

  /// \brief Create an unknown-valued ciphertext

  /// \brief Create ciphertext wrapper from ciphertext
  /// \param[in] cipher Ciphertext to store
  /// TODO: add move constructor
  SealCiphertextWrapper(const seal::Ciphertext& cipher)
      : m_ciphertext(cipher) {}

  /// \brief Returns the underyling SEAL ciphertext
  inline seal::Ciphertext& ciphertext() { return m_ciphertext; }

  /// \brief Returns the underyling SEAL ciphertext
  inline const seal::Ciphertext& ciphertext() const { return m_ciphertext; }

  /// \brief Returns the size of the underlying ciphertext
  inline size_t size() const { return m_ciphertext.size(); }

  /// \brief Returns scale of the ciphertext
  inline double& scale() { return m_ciphertext.scale(); }

  /// \brief Returns scale of the ciphertext
  inline double scale() const { return m_ciphertext.scale(); }

  /// \brief Saves the ciphertext to a protobuf ciphertext wrapper
  /// \param[out] proto_cipher Protobuf ciphertext wrapper to store the
  /// ciphertext
  inline void save(he_proto::SealCiphertextWrapper& proto_cipher) const {
    // proto_cipher.set_complex_packing(complex_packing());

    size_t cipher_size = ciphertext_size(m_ciphertext);
    std::string cipher_str;
    cipher_str.resize(cipher_size);

    size_t save_size = ngraph::he::save(
        m_ciphertext, reinterpret_cast<std::byte*>(cipher_str.data()));

    NGRAPH_CHECK(save_size == cipher_size, "Save size != cipher size");

    proto_cipher.set_ciphertext(std::move(cipher_str));
  }

  /// \brief Loads a ciphertext from a buffer to a SealCiphertextWrapper
  /// \param[out] dst Destination to load ciphertext wrapper to
  /// \param[in] src Source to load ciphertext wrapper from
  /// \param[in] context SEAL context of ciphertext to load
  static inline void load(SealCiphertextWrapper& dst,
                          const he_proto::SealCiphertextWrapper& src,
                          std::shared_ptr<seal::SEALContext> context) {
    // dst.complex_packing() = src.complex_packing();

    // TODO: load from string directly
    const std::string& cipher_str = src.ciphertext();
    ngraph::he::load(dst.ciphertext(), context,
                     reinterpret_cast<const std::byte*>(cipher_str.data()),
                     cipher_str.size());
  }

  /// \brief Loads a ciphertext from a buffer to a SealCiphertextWrapper
  /// \param[out] dst Destination to load ciphertext wrapper to
  /// \param[in] src Source to load ciphertext wrapper from
  /// \param[in] context SEAL context of ciphertext to load
  static inline void load(std::shared_ptr<SealCiphertextWrapper>& dst,
                          const he_proto::SealCiphertextWrapper& src,
                          std::shared_ptr<seal::SEALContext> context) {
    dst = std::make_shared<SealCiphertextWrapper>();
    load(*dst, src, context);
  }

 private:
  seal::Ciphertext m_ciphertext;
  bool m_complex_packing;
};

}  // namespace he
}  // namespace ngraph
