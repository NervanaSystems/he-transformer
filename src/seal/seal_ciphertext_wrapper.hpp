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

namespace ngraph {
namespace he {
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
    std::stringstream s;
    m_ciphertext.save(s);
    proto_cipher.set_ciphertext(s.str());
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
      std::stringstream ss;
      ss.str(cipher_str);
      dst.ciphertext().load(context, ss);
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

}  // namespace he
}  // namespace ngraph
