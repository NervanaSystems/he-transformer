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

#include "protos/message.pb.h"
#include "seal/seal.h"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {
/// \brief Returns the size in bytes required to serialize a ciphertext
/// \param[in] cipher Ciphertext to measure size of
inline size_t ciphertext_size(const seal::Ciphertext& cipher) {
  // TODO: figure out why the extra 8 bytes
  size_t expected_size = 8;
  expected_size += sizeof(seal::parms_id_type);
  expected_size += sizeof(seal::SEAL_BYTE);
  // size64, poly_modulus_degere, coeff_mod_count
  expected_size += 3 * sizeof(uint64_t);
  // scale
  expected_size += sizeof(double);
  // data
  expected_size += 8 * cipher.uint64_count();
  return expected_size;
}

/// \brief Serializes the ciphertext and writes to a destination
/// \param[in] cipher Ciphertext to write
/// \param[out] destination Where to save ciphertext to
inline void save(const seal::Ciphertext& cipher, void* destination) {
  {
    static constexpr std::array<size_t, 6> offsets = {
        sizeof(seal::parms_id_type),
        sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE),
        sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
            sizeof(uint64_t),
        sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
            2 * sizeof(uint64_t),
        sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
            3 * sizeof(uint64_t),
        sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
            3 * sizeof(uint64_t) + sizeof(double),
    };

    bool is_ntt_form = cipher.is_ntt_form();
    uint64_t size = cipher.size();
    uint64_t polynomial_modulus_degree = cipher.poly_modulus_degree();
    uint64_t coeff_mod_count = cipher.coeff_mod_count();

    char* dst_char = static_cast<char*>(destination);
    std::memcpy(destination, (void*)&cipher.parms_id(),
                sizeof(seal::parms_id_type));
    std::memcpy(static_cast<void*>(dst_char + offsets[0]), (void*)&is_ntt_form,
                sizeof(seal::SEAL_BYTE));
    std::memcpy(static_cast<void*>(dst_char + offsets[1]), (void*)&size,
                sizeof(uint64_t));
    std::memcpy(static_cast<void*>(dst_char + offsets[2]),
                (void*)&polynomial_modulus_degree, sizeof(uint64_t));
    std::memcpy(static_cast<void*>(dst_char + offsets[3]),
                (void*)&coeff_mod_count, sizeof(uint64_t));
    std::memcpy(static_cast<void*>(dst_char + offsets[4]),
                (void*)&cipher.scale(), sizeof(double));
    std::memcpy(static_cast<void*>(dst_char + offsets[5]), (void*)cipher.data(),
                8 * cipher.uint64_count());
  }
}

/// \brief Loads a serialized ciphertext
/// \param[out] cipher De-serialized ciphertext
/// \param[in] context Encryption context to verify ciphertext validity against
/// \param[in] src Pointer to data to load from
inline void load(seal::Ciphertext& cipher,
                 std::shared_ptr<seal::SEALContext> context, void* src) {
  seal::SEAL_BYTE is_ntt_form_byte;
  uint64_t size64 = 0;
  seal::parms_id_type parms_id{};

  static constexpr std::array<size_t, 6> offsets = {
      sizeof(seal::parms_id_type),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) + sizeof(uint64_t),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
          2 * sizeof(uint64_t),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
          3 * sizeof(uint64_t),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
          3 * sizeof(uint64_t) + sizeof(double),
  };

  char* char_src = static_cast<char*>(src);
  std::memcpy(&parms_id, src, sizeof(seal::parms_id_type));

  seal::Ciphertext new_cipher(context, parms_id);

  std::memcpy(&is_ntt_form_byte, static_cast<void*>(char_src + offsets[0]),
              sizeof(seal::SEAL_BYTE));
  std::memcpy(&size64, static_cast<void*>(char_src + offsets[1]),
              sizeof(uint64_t));
  std::memcpy(&new_cipher.scale(), static_cast<void*>(char_src + offsets[4]),
              sizeof(double));
  bool ntt_form = (is_ntt_form_byte == seal::SEAL_BYTE(0)) ? false : true;

  new_cipher.resize(context, parms_id, size64);

  new_cipher.is_ntt_form() = ntt_form;
  void* data_src = static_cast<void*>(char_src + offsets[5]);
  std::memcpy(&new_cipher[0], data_src,
              new_cipher.uint64_count() * sizeof(std::uint64_t));
  cipher = std::move(new_cipher);

  NGRAPH_CHECK(seal::is_valid_for(cipher, context),
               "ciphertext data is invalid");
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
  seal::Ciphertext& ciphertext() { return m_ciphertext; }

  /// \brief Returns the underyling SEAL ciphertext
  const seal::Ciphertext& ciphertext() const { return m_ciphertext; }

  /// \brief Serializes the ciphertext to a stream
  /// \param[out] Stream to serialize the ciphertext to
  void save(std::ostream& stream) const { m_ciphertext.save(stream); }

  /// \brief Returns the size of the underlying ciphertext
  size_t size() const { return m_ciphertext.size(); }

  /// \brief Returns whether or not ciphertext represents a known value
  bool known_value() const { return m_known_value; }

  /// \brief Returns whether or not ciphertext represents a known value
  bool& known_value() { return m_known_value; }

  /// \brief Returns known value
  float value() const { return m_value; }

  /// \brief Returns known value
  float& value() { return m_value; }

  /// \brief Returns scale of the ciphertext
  double& scale() { return m_ciphertext.scale(); }

  /// \brief Returns scale of the ciphertext
  double scale() const { return m_ciphertext.scale(); }

  /// \brief Returns whether or not the ciphertext uses complex packing
  bool complex_packing() const { return m_complex_packing; }

  /// \brief Returns whether or not the ciphertext uses complex packing
  bool& complex_packing() { return m_complex_packing; }

  /// \brief Saves the cihertext to a protobuf ciphertext wrapper
  /// \param[out] proto_cipher Protobuf ciphertext wrapper to store the
  /// ciphertext
  inline void save(he_proto::SealCiphertextWrapper& proto_cipher) const {
    proto_cipher.set_complex_packing(complex_packing());
    proto_cipher.set_known_value(known_value());
    if (known_value()) {
      proto_cipher.set_value(value());
    }

    // TODO: save directly to protobuf
    size_t stream_size = ngraph::he::ciphertext_size(m_ciphertext);
    std::string cipher_str;
    cipher_str.resize(stream_size);
    ngraph::he::save(m_ciphertext, cipher_str.data());
    proto_cipher.set_ciphertext(std::move(cipher_str));
  }

  /// \brief Loads a ciphertext from a buffer to a SealCiphertextWrapper
  /// \param[out] dst Destination to load ciphertext wrapper to
  /// \param[in] src Source to load ciphertext wrapper from
  /// \param[in] context TODO
  static inline void load(ngraph::he::SealCiphertextWrapper& dst,
                          const he_proto::SealCiphertextWrapper& src,
                          std::shared_ptr<seal::SEALContext> context) {
    dst.complex_packing() = src.complex_packing();
    if (src.known_value()) {
      dst.known_value() = true;
      dst.value() = src.value();
    } else {
      // TODO: load from string directly
      const std::string& cipher_str = src.ciphertext();
      ngraph::he::load(
          dst.ciphertext(), context,
          static_cast<void*>(const_cast<char*>(cipher_str.data())));
    }
  }

  /// \brief Loads a ciphertext from a buffer to a SealCiphertextWrapper
  /// \param[out] dst Destination to load ciphertext wrapper to
  /// \param[in] src Source to load ciphertext wrapper from
  /// \param[in] context TODO
  static inline void load(
      std::shared_ptr<ngraph::he::SealCiphertextWrapper>& dst,
      const he_proto::SealCiphertextWrapper& src,
      std::shared_ptr<seal::SEALContext> context) {
    dst = std::make_shared<ngraph::he::SealCiphertextWrapper>();
    load(*dst, src, context);
  }

 private:
  seal::Ciphertext m_ciphertext;
  bool m_complex_packing;
  bool m_known_value;
  float m_value{0.0f};
};

/// \brief Saves a list of ciphertexts to a protobuf message
/// \param[in] ciphers_begin Iterator at beginning of ciphertext wrapper list
/// \param[in] ciphers_end Iterator at end of ciphertext wrapper list
/// \param[out] proto_msg Protobuf message to save ciphertexts to
inline void save_to_proto(
    std::vector<std::shared_ptr<SealCiphertextWrapper>>::const_iterator
        ciphers_begin,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>::const_iterator
        ciphers_end,
    he_proto::TCPMessage& proto_msg) {
  size_t ciphers_size = 0;
  for (auto it = ciphers_begin; it != ciphers_end; ++it) {
    proto_msg.add_ciphers();
    ciphers_size++;
  }
#pragma omp parallel for
  for (size_t cipher_idx = 0; cipher_idx < ciphers_size; ++cipher_idx) {
    auto cipher_wrapper = *(ciphers_begin + cipher_idx);
    cipher_wrapper->save(*proto_msg.mutable_ciphers(cipher_idx));
  }
}

/// \brief Saves a vector of ciphertexts to a protobuf message
/// \param[in] ciphers Vector of ciphertext wrappers to save
/// \param[out] proto_msg Protobuf message to save ciphertexts to
inline void save_to_proto(
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& ciphers,
    he_proto::TCPMessage& proto_msg) {
  proto_msg.mutable_ciphers()->Reserve(ciphers.size());
  for (size_t cipher_idx = 0; cipher_idx < ciphers.size(); ++cipher_idx) {
    proto_msg.add_ciphers();
  }
#pragma omp parallel for
  for (size_t cipher_idx = 0; cipher_idx < ciphers.size(); ++cipher_idx) {
    ciphers[cipher_idx]->save(*proto_msg.mutable_ciphers(cipher_idx));
  }
}

}  // namespace he
}  // namespace ngraph
