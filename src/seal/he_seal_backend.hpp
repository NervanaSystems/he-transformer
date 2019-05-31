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

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "he_encryption_parameters.hpp"
#include "he_plaintext.hpp"
#include "he_tensor.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/performance_counter.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"
#include "node_wrapper.hpp"
#include "seal/he_seal_encryption_parameters.hpp"
#include "seal/he_seal_executable.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace he {
class HETensor;
class HESealBackend : public ngraph::runtime::Backend {
 public:
  //
  // ngraph backend overrides
  //
  std::shared_ptr<runtime::Tensor> create_tensor(
      const element::Type& element_type, const Shape& shape) override;

  inline std::shared_ptr<runtime::Tensor> create_tensor(
      const element::Type& element_type, const Shape& shape,
      void* memory_pointer) override {
    throw ngraph_error("create_tensor unimplemented");
  }

  std::shared_ptr<ngraph::runtime::Executable> compile(
      std::shared_ptr<Function> func,
      bool enable_performance_data = false) override;

  void validate_he_call(std::shared_ptr<const Function> function,
                        const std::vector<std::shared_ptr<HETensor>>& outputs,
                        const std::vector<std::shared_ptr<HETensor>>& inputs);

  //
  // Tensor creation
  //
  std::shared_ptr<runtime::Tensor> create_batched_cipher_tensor(
      const element::Type& element_type, const Shape& shape);

  std::shared_ptr<runtime::Tensor> create_batched_plain_tensor(
      const element::Type& element_type, const Shape& shape);

  std::shared_ptr<runtime::Tensor> create_plain_tensor(
      const element::Type& element_type, const Shape& shape,
      const bool batched = false) const;

  /// @brief Constructs SEAL context from SEAL parameter
  /// @param sp SEAL Parameter from which to construct context
  /// @return Pointer to constructed context
  std::shared_ptr<seal::SEALContext> make_seal_context(
      const std::shared_ptr<ngraph::he::HEEncryptionParameters> sp);

  std::shared_ptr<ngraph::he::SealCiphertextWrapper> create_empty_ciphertext()
      const;

  std::shared_ptr<ngraph::he::SealCiphertextWrapper> create_empty_ciphertext(
      seal::parms_id_type parms_id) const;

  std::shared_ptr<ngraph::he::SealCiphertextWrapper> create_empty_ciphertext(
      const seal::MemoryPoolHandle& pool) const;

  /// @brief Creates ciphertext of unspecified value
  /// Alias for create_empty_ciphertext()
  /// @return Shared pointer to created ciphertext
  template <typename T,
            typename = std::enable_if_t<
                std::is_same<T, ngraph::he::SealCiphertextWrapper>::value ||
                std::is_same<T, std::shared_ptr<
                                    ngraph::he::SealCiphertextWrapper>>::value>>
  std::shared_ptr<ngraph::he::SealCiphertextWrapper> create_empty_hetext()
      const {
    return create_empty_ciphertext();
  };

  /// @brief Creates plaintext of unspecified value using memory pool
  /// Alias for create_empty_plaintext()
  /// @return Shared pointer to created plaintext
  template <typename T, typename = std::enable_if_t<
                            std::is_same<T, ngraph::he::HEPlaintext>::value>>
  std::unique_ptr<ngraph::he::HEPlaintext> create_empty_hetext() const {
    return ngraph::he::create_empty_plaintext();
  };

  void encode(ngraph::he::SealPlaintextWrapper& destination,
              const ngraph::he::HEPlaintext& plaintext,
              seal::parms_id_type parms_id, double scale) const;

  void encode(ngraph::he::SealPlaintextWrapper& destination,
              const ngraph::he::HEPlaintext& plaintext) const;

  void encode(ngraph::he::HEPlaintext& output, const void* input,
              const element::Type& type, bool complex, size_t count = 1) const;

  void decode(void* output, const ngraph::he::HEPlaintext& input,
              const element::Type& type, size_t count = 1) const;

  void decode(ngraph::he::HEPlaintext& output,
              const ngraph::he::SealPlaintextWrapper& input) const;

  void encrypt(std::shared_ptr<ngraph::he::SealCiphertextWrapper>& output,
               const ngraph::he::HEPlaintext& input) const;

  void decrypt(
      ngraph::he::HEPlaintext& output,
      const std::shared_ptr<ngraph::he::SealCiphertextWrapper>& input) const;

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

  const std::shared_ptr<ngraph::he::HEEncryptionParameters>
  get_encryption_parameters() const {
    return m_encryption_params;
  };

  void set_batch_data(bool batch) { m_batch_data = batch; };
  void set_complex_packing(bool toggle) { m_complex_packing = toggle; }

  bool encrypt_data() const { return m_encrypt_data; };
  bool batch_data() const { return m_batch_data; };
  bool encrypt_model() const { return m_encrypt_model; };
  bool complex_packing() const { return m_complex_packing; };

 private:
  bool m_encrypt_data{std::getenv("NGRAPH_ENCRYPT_DATA") != nullptr};
  bool m_batch_data{std::getenv("NGRAPH_BATCH_DATA") != nullptr};
  bool m_encrypt_model{std::getenv("NGRAPH_ENCRYPT_MODEL") != nullptr};
  bool m_complex_packing{std::getenv("NGRAPH_COMPLEX_PACK") != nullptr};

  std::shared_ptr<seal::SecretKey> m_secret_key;
  std::shared_ptr<seal::PublicKey> m_public_key;
  std::shared_ptr<seal::RelinKeys> m_relin_keys;
  std::shared_ptr<seal::Encryptor> m_encryptor;
  std::shared_ptr<seal::Decryptor> m_decryptor;
  std::shared_ptr<seal::SEALContext> m_context;
  std::shared_ptr<seal::Evaluator> m_evaluator;
  std::shared_ptr<seal::KeyGenerator> m_keygen;
  std::shared_ptr<HEEncryptionParameters> m_encryption_params;
};

}  // namespace he
}  // namespace ngraph
