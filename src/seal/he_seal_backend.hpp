//*****************************************************************************
// Copyright 2018 Intel Corporation
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
#include "ngraph/runtime/backend.hpp"
#include "seal/he_seal_parameter.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

#include "seal/seal.h"

namespace ngraph {
namespace runtime {
namespace he {
class HETensor;
class HEPlainTensor;
class HECipherTensor;
class HEBackend;

namespace he_seal {
class HESealBackend : public HEBackend {
 public:
  ~HESealBackend(){};

  /// @brief Constructs SEAL context from SEAL parameter
  /// @param sp SEAL Parameter from which to construct context
  /// @return Pointer to constructed context
  virtual std::shared_ptr<seal::SEALContext> make_seal_context(
      const std::shared_ptr<runtime::he::he_seal::HESealParameter> sp)
      const = 0;

  /// @brief Checks if parameter is valid for encoding.
  ///        Throws an error if parameter is not valid.
  void assert_valid_seal_parameter(
      const std::shared_ptr<runtime::he::he_seal::HESealParameter> sp) const;

  virtual std::shared_ptr<runtime::Tensor> create_batched_tensor(
      const element::Type& element_type, const Shape& shape) = 0;

  std::shared_ptr<runtime::he::HECiphertext> create_empty_ciphertext()
      const override;

  std::shared_ptr<runtime::he::HECiphertext> create_empty_ciphertext(
      const seal::MemoryPoolHandle& pool) const;

  std::shared_ptr<runtime::he::HEPlaintext> create_empty_plaintext()
      const override;

  std::shared_ptr<runtime::he::HEPlaintext> create_empty_plaintext(
      const seal::MemoryPoolHandle& pool) const;

  /// @brief Creates ciphertext of unspecified value using memory pool
  /// Alias for create_empty_ciphertext()
  /// @return Shared pointer to created ciphertext
  template <typename T>
  std::shared_ptr<runtime::he::HECiphertext> create_empty_hetext(
      runtime::he::HECiphertext&&, const seal::MemoryPoolHandle& pool) const {
    return create_empty_ciphertext(pool);
  };

  /// @brief Creates plaintext of unspecified value using memory pool
  /// Alias for create_empty_plaintext()
  /// @return Shared pointer to created plaintext
  template <typename T>
  std::shared_ptr<runtime::he::HEPlaintext> create_empty_hetext(
      runtime::he::HEPlaintext&&, const seal::MemoryPoolHandle& pool) const {
    return create_empty_plaintext(pool);
  };

  virtual void encode(std::shared_ptr<runtime::he::HEPlaintext>& output,
                      const void* input, const element::Type& element_type,
                      size_t count = 1) const = 0;

  virtual void decode(void* output,
                      const std::shared_ptr<runtime::he::HEPlaintext> input,
                      const element::Type& element_type,
                      size_t count = 1) const = 0;

  void encrypt(
      std::shared_ptr<runtime::he::HECiphertext>& output,
      const std::shared_ptr<runtime::he::HEPlaintext> input) const override;

  void decrypt(
      std::shared_ptr<runtime::he::HEPlaintext>& output,
      const std::shared_ptr<runtime::he::HECiphertext> input) const override;

  const inline std::shared_ptr<seal::SEALContext> get_context() const {
    return m_context;
  }

  const inline std::shared_ptr<seal::SecretKey> get_secret_key() const {
    return m_secret_key;
  }

  const inline std::shared_ptr<seal::RelinKeys> get_relin_keys() const {
    return m_relin_keys;
  }

  const inline std::shared_ptr<seal::Evaluator> get_evaluator() const {
    return m_evaluator;
  }

  const std::shared_ptr<const runtime::he::HEPlaintext> get_valued_plaintext(
      double value) const;

 protected:
  std::shared_ptr<seal::SecretKey> m_secret_key;
  std::shared_ptr<seal::PublicKey> m_public_key;
  std::shared_ptr<seal::RelinKeys> m_relin_keys;
  std::shared_ptr<seal::Encryptor> m_encryptor;
  std::shared_ptr<seal::Decryptor> m_decryptor;
  std::shared_ptr<seal::SEALContext> m_context;
  std::shared_ptr<seal::Evaluator> m_evaluator;
  std::shared_ptr<seal::KeyGenerator> m_keygen;

  std::unordered_map<double, std::shared_ptr<runtime::he::HEPlaintext>>
      m_plaintext_map;
};
}  // namespace he_seal
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
