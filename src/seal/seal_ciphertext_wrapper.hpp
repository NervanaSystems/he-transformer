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
  /// \brief Create an empty unknown-valued ciphertext without complex packing
  SealCiphertextWrapper() : m_complex_packing(false), m_known_value(false) {}

  /// \brief Create an unknown-valued ciphertext
  /// \param[in] complex_packign Whether or not to use complex packing
  SealCiphertextWrapper(bool complex_packing)
      : m_complex_packing(complex_packing), m_known_value(false) {}

  /// \brief Create ciphertext wrapper from ciphertext
  /// \param[in] cipher Ciphertext to store
  /// \param[in] complex_packing Whether or not ciphertext uses complex packing
  /// TODO: add move constructor
  SealCiphertextWrapper(const seal::Ciphertext& cipher,
                        bool complex_packing = false)
      : m_ciphertext(cipher),
        m_complex_packing(complex_packing),
        m_known_value(false) {}

  /// \brief Returns the underyling SEAL ciphertext
  inline seal::Ciphertext& ciphertext() { return m_ciphertext; }

  /// \brief Returns the underyling SEAL ciphertext
  inline const seal::Ciphertext& ciphertext() const { return m_ciphertext; }

  /// \brief Serializes the ciphertext to a stream
  /// \param[out] Stream to serialize the ciphertext to
  inline void save(std::ostream& stream) const { m_ciphertext.save(stream); }

  /// \brief Returns the size of the underlying ciphertext
  inline size_t size() const { return m_ciphertext.size(); }

  /// \brief Returns whether or not ciphertext represents a known value
  inline bool known_value() const { return m_known_value; }

  /// \brief Returns whether or not ciphertext represents a known value
  inline bool& known_value() { return m_known_value; }

  /// \brief Returns known value
  inline float value() const { return m_value; }

  /// \brief Returns known value
  inline float& value() { return m_value; }

  /// \brief Returns scale of the ciphertext
  inline double& scale() { return m_ciphertext.scale(); }

  /// \brief Returns scale of the ciphertext
  inline double scale() const { return m_ciphertext.scale(); }

  /// \brief Returns whether or not the ciphertext uses complex packing
  inline bool complex_packing() const { return m_complex_packing; }

  /// \brief Returns whether or not the ciphertext uses complex packing
  inline bool& complex_packing() { return m_complex_packing; }

  /// \brief Saves the cihertext to a protobuf ciphertext wrapper
  /// \param[out] proto_cipher Protobuf ciphertext wrapper to store the
  /// ciphertext
  inline void save(he_proto::SealCiphertextWrapper& proto_cipher) const {
    using Clock = std::chrono::high_resolution_clock;
    auto t1 = Clock::now();
    proto_cipher.set_complex_packing(complex_packing());
    proto_cipher.set_known_value(known_value());
    if (known_value()) {
      proto_cipher.set_value(value());
    }

    // New method
    if (true) {
      size_t cipher_size = ciphertext_size(m_ciphertext);
      std::string cipher_str;
      cipher_str.resize(cipher_size);

      auto t2 = Clock::now();

      size_t save_size = ngraph::he::save(
          m_ciphertext, reinterpret_cast<std::byte*>(cipher_str.data()));

      auto t3 = Clock::now();
      proto_cipher.set_ciphertext(std::move(cipher_str));

      auto t4 = Clock::now();
      auto t5 = Clock::now();

      NGRAPH_INFO << "cipher_size " << cipher_size;
      NGRAPH_INFO << "save_size " << save_size;

      NGRAPH_HE_LOG(3) << "malloc took "
                       << std::chrono::duration_cast<std::chrono::microseconds>(
                              t2 - t1)
                              .count()
                       << "us";
      NGRAPH_HE_LOG(3) << "save took "
                       << std::chrono::duration_cast<std::chrono::microseconds>(
                              t3 - t2)
                              .count()
                       << "us";
      NGRAPH_HE_LOG(3) << "set_ciphertext took "
                       << std::chrono::duration_cast<std::chrono::microseconds>(
                              t4 - t3)
                              .count()
                       << "us";
      NGRAPH_HE_LOG(3) << "ngraph_free took "
                       << std::chrono::duration_cast<std::chrono::microseconds>(
                              t5 - t4)
                              .count()
                       << "us";
      NGRAPH_HE_LOG(3) << "Total save took "
                       << std::chrono::duration_cast<std::chrono::microseconds>(
                              t5 - t1)
                              .count()
                       << "us with save size " << save_size;
      NGRAPH_INFO << "Saved ciphertext size "
                  << proto_cipher.ciphertext().size();

    } else {  // Old method
      size_t cipher_size = ciphertext_size(m_ciphertext);

      std::byte* cipher_data =
          static_cast<std::byte*>(ngraph::ngraph_malloc(cipher_size));

      auto t2 = Clock::now();

      size_t save_size = ngraph::he::save(m_ciphertext, cipher_data);

      auto t3 = Clock::now();
      proto_cipher.set_ciphertext(static_cast<void*>(cipher_data), save_size);

      auto t4 = Clock::now();

      ngraph::ngraph_free(cipher_data);

      auto t5 = Clock::now();

      NGRAPH_HE_LOG(3) << "malloc took "
                       << std::chrono::duration_cast<std::chrono::microseconds>(
                              t2 - t1)
                              .count()
                       << "us";
      NGRAPH_HE_LOG(3) << "save took "
                       << std::chrono::duration_cast<std::chrono::microseconds>(
                              t3 - t2)
                              .count()
                       << "us";
      NGRAPH_HE_LOG(3) << "set_ciphertext took "
                       << std::chrono::duration_cast<std::chrono::microseconds>(
                              t4 - t3)
                              .count()
                       << "us";
      NGRAPH_HE_LOG(3) << "ngraph_free took "
                       << std::chrono::duration_cast<std::chrono::microseconds>(
                              t5 - t4)
                              .count()
                       << "us";
      NGRAPH_HE_LOG(3) << "Total save took "
                       << std::chrono::duration_cast<std::chrono::microseconds>(
                              t5 - t1)
                              .count()
                       << "us with save size " << save_size;
    }
    NGRAPH_INFO << "Saved ciphertext (out of scope) size "
                << proto_cipher.ciphertext().size();
  }

  /// \brief Loads a ciphertext from a buffer to a SealCiphertextWrapper
  /// \param[out] dst Destination to load ciphertext wrapper to
  /// \param[in] src Source to load ciphertext wrapper from
  /// \param[in] context SEAL context of ciphertext to load
  static inline void load(SealCiphertextWrapper& dst,
                          const he_proto::SealCiphertextWrapper& src,
                          std::shared_ptr<seal::SEALContext> context) {
    dst.complex_packing() = src.complex_packing();
    if (src.known_value()) {
      dst.known_value() = true;
      dst.value() = src.value();
    } else {
      // TODO: load from string directly
      const std::string& cipher_str = src.ciphertext();
      ngraph::he::load(dst.ciphertext(), context,
                       reinterpret_cast<const std::byte*>(cipher_str.data()),
                       cipher_str.size());
    }
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
  bool m_known_value;
  float m_value{0.0f};
};

}  // namespace he
}  // namespace ngraph
