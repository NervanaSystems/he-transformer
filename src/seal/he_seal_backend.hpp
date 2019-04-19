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
#include <unordered_map>

#include "he_backend.hpp"
#include "seal/he_seal_encryption_parameters.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace runtime {
namespace he {
namespace he_seal {
class HESealBackend : public HEBackend {
 public:
  /// @brief Constructs SEAL context from SEAL parameter
  /// @param sp SEAL Parameter from which to construct context
  /// @return Pointer to constructed context
  virtual std::shared_ptr<seal::SEALContext> make_seal_context(
      const std::shared_ptr<runtime::he::HEEncryptionParameters> sp) = 0;

  virtual ~HESealBackend(){};

  virtual std::shared_ptr<runtime::Tensor> create_batched_cipher_tensor(
      const element::Type& type, const Shape& shape) override = 0;

  virtual std::shared_ptr<runtime::Tensor> create_batched_plain_tensor(
      const element::Type& type, const Shape& shape) override = 0;

  std::shared_ptr<runtime::he::HECiphertext> create_empty_ciphertext()
      const override;

  std::shared_ptr<runtime::he::HECiphertext> create_empty_ciphertext(
      seal::parms_id_type parms_id) const;

  std::shared_ptr<runtime::he::HECiphertext> create_empty_ciphertext(
      const seal::MemoryPoolHandle& pool) const;

  std::shared_ptr<runtime::he::HEPlaintext> create_empty_plaintext()
      const override;

  std::shared_ptr<runtime::he::HEPlaintext> create_empty_plaintext(
      const seal::MemoryPoolHandle& pool) const;

  /// @brief Creates ciphertext of unspecified value using memory pool
  /// Alias for create_empty_ciphertext()
  /// @return Shared pointer to created ciphertext
  template <typename T, typename = std::enable_if_t<
                            std::is_same<T, runtime::he::HECiphertext>::value>>
  std::shared_ptr<runtime::he::HECiphertext> create_empty_hetext(
      const seal::MemoryPoolHandle& pool) const {
    return create_empty_ciphertext(pool);
  };

  /// @brief Creates plaintext of unspecified value using memory pool
  /// Alias for create_empty_plaintext()
  /// @return Shared pointer to created plaintext
  template <typename T, typename = std::enable_if_t<
                            std::is_same<T, runtime::he::HEPlaintext>::value>>
  std::shared_ptr<runtime::he::HEPlaintext> create_empty_hetext(
      const seal::MemoryPoolHandle& pool) const {
    return create_empty_plaintext(pool);
  };

  virtual void encode(
      runtime::he::he_seal::SealPlaintextWrapper* plaintext) = 0;

  virtual void encode(std::shared_ptr<runtime::he::HEPlaintext>& output,
                      const void* input, const element::Type& type,
                      size_t count = 1) const = 0;

  virtual void decode(void* output, const runtime::he::HEPlaintext* input,
                      const element::Type& type, size_t count = 1) const = 0;

  void encrypt(std::shared_ptr<runtime::he::HECiphertext>& output,
               runtime::he::HEPlaintext* input) override;

  void decrypt(std::shared_ptr<runtime::he::HEPlaintext>& output,
               const runtime::he::HECiphertext* input) const override;

  const inline std::shared_ptr<seal::SEALContext> get_context() const noexcept {
    return m_context;
  }

  const inline std::shared_ptr<seal::SecretKey> get_secret_key() const
      noexcept {
    return m_secret_key;
  }

  const inline std::shared_ptr<seal::PublicKey> get_public_key() const
      noexcept {
    return m_public_key;
  }

  const inline std::shared_ptr<seal::RelinKeys> get_relin_keys() const
      noexcept {
    return m_relin_keys;
  }

  void set_relin_keys(const seal::RelinKeys& keys) {
    m_relin_keys = std::make_shared<seal::RelinKeys>(keys);
  }

  void set_public_key(const seal::PublicKey& key) {
    m_public_key = std::make_shared<seal::PublicKey>(key);
    m_encryptor = std::make_shared<seal::Encryptor>(m_context, *m_public_key);
  }

  const inline std::shared_ptr<seal::Evaluator> get_evaluator() const noexcept {
    return m_evaluator;
  }

 protected:
  std::shared_ptr<seal::SecretKey> m_secret_key;
  std::shared_ptr<seal::PublicKey> m_public_key;
  std::shared_ptr<seal::RelinKeys> m_relin_keys;
  std::shared_ptr<seal::Encryptor> m_encryptor;
  std::shared_ptr<seal::Decryptor> m_decryptor;
  std::shared_ptr<seal::SEALContext> m_context;
  std::shared_ptr<seal::Evaluator> m_evaluator;
  std::shared_ptr<seal::KeyGenerator> m_keygen;
};
}  // namespace he_seal
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
