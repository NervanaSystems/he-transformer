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

  if (!seal::is_valid_for(cipher, context)) {
    throw std::invalid_argument("ciphertext data is invalid");
  }
}

class SealCiphertextWrapper {
 public:
  SealCiphertextWrapper() : m_complex_packing(false), m_known_value(false) {}

  SealCiphertextWrapper(bool complex_packing)
      : m_complex_packing(complex_packing), m_known_value(false) {}

  SealCiphertextWrapper(const seal::Ciphertext& cipher,
                        bool complex_packing = false, bool known_value = false)
      : m_ciphertext(cipher),
        m_complex_packing(complex_packing),
        m_known_value(known_value) {}

  seal::Ciphertext& ciphertext() { return m_ciphertext; }
  const seal::Ciphertext& ciphertext() const { return m_ciphertext; }

  void save(std::ostream& stream) const { m_ciphertext.save(stream); }

  size_t size() const { return m_ciphertext.size(); }

  bool known_value() const { return m_known_value; }
  bool& known_value() { return m_known_value; }

  float value() const { return m_value; }
  float& value() { return m_value; }

  double& scale() { return m_ciphertext.scale(); }
  double scale() const { return m_ciphertext.scale(); }

  bool complex_packing() const { return m_complex_packing; }
  bool& complex_packing() { return m_complex_packing; }

  inline void save(he_proto::SealCiphertextWrapper& proto_cipher) {
    proto_cipher.set_known_value(known_value());
    if (known_value()) {
      proto_cipher.set_value(value());
    }
    proto_cipher.set_complex_packing(complex_packing());

    // TODO: save directly to protobuf
    size_t stream_size = ngraph::he::ciphertext_size(m_ciphertext);
    std::string cipher_str;
    cipher_str.resize(stream_size);
    ngraph::he::save(m_ciphertext, cipher_str.data());
    proto_cipher.set_ciphertext(std::move(cipher_str));
  }

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
  float m_value;
};

inline void save_to_proto(
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& ciphers,
    he_proto::TCPMessage& proto_msg) {
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
