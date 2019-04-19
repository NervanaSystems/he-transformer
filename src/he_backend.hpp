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

#include "he_ciphertext.hpp"
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

namespace ngraph {
namespace runtime {
namespace he {
class HETensor;
class HEBackend : public runtime::Backend {
 public:
  virtual ~HEBackend(){};

  /// @brief Creates ciphertext of unspecified value
  /// @return Shared pointer to created ciphertext
  virtual std::shared_ptr<runtime::he::HECiphertext> create_empty_ciphertext()
      const = 0;

  /// @brief Creates plaintext of unspecified value
  /// @return Shared pointer to created plaintext
  virtual std::shared_ptr<runtime::he::HEPlaintext> create_empty_plaintext()
      const = 0;

  /// @brief Creates ciphertext of unspecified value
  /// Alias for create_empty_ciphertext()
  /// @return Shared pointer to created ciphertext
  template <typename T, typename = std::enable_if_t<
                            std::is_same<T, runtime::he::HECiphertext>::value>>
  std::shared_ptr<runtime::he::HECiphertext> create_empty_hetext() {
    return create_empty_ciphertext();
  };

  /// @brief Creates plaintext of unspecified value
  /// Alias for create_empty_plaintext()
  /// @return Shared pointer to created plaintext
  template <typename T, typename = std::enable_if_t<
                            std::is_same<T, runtime::he::HEPlaintext>::value>>
  std::shared_ptr<runtime::he::HEPlaintext> create_empty_hetext() {
    return create_empty_plaintext();
  };

  /// @brief Creates ciphertext of specified value
  /// @param value Scalar which to encrypt
  /// @param element_type Type to encrypt
  /// @param batch_size Number of elements to encrypt
  ///        > 1 indicates batching
  /// @return Shared pointer to created ciphertext
  std::shared_ptr<runtime::he::HECiphertext> create_valued_ciphertext(
      float value, const element::Type& element_type, size_t batch_size = 1);

  /// @brief Creates plaintext of specified value
  /// @param value Scalar which to encode
  /// @param element_type Type to encode
  /// @return Shared pointer to created plaintext
  std::shared_ptr<runtime::he::HEPlaintext> create_valued_plaintext(
      float value, const element::Type& element_type);

  /// @brief Creates ciphertext of specified value
  /// Alias for create_valued_ciphertext()
  /// @return Shared pointer to created ciphertext
  template <typename T, typename = std::enable_if_t<
                            std::is_same<T, runtime::he::HECiphertext>::value>>
  std::shared_ptr<runtime::he::HECiphertext> create_valued_hetext(
      float value, const element::Type& element_type, size_t batch_size = 1) {
    return create_valued_ciphertext(value, element_type, batch_size);
  };

  /// @brief Creates plaintext of specified value
  /// Alias for create_valued_plaintext()
  /// @return Shared pointer to created plaintext
  template <typename T, typename = std::enable_if_t<
                            std::is_same<T, runtime::he::HEPlaintext>::value>>
  std::shared_ptr<runtime::he::HEPlaintext> create_valued_hetext(
      float value, const element::Type& element_type) {
    return create_valued_plaintext(value, element_type);
  };

  std::shared_ptr<runtime::Tensor> create_tensor(
      const element::Type& element_type, const Shape& shape) override;

  virtual std::shared_ptr<runtime::Tensor> create_batched_cipher_tensor(
      const element::Type& element_type, const Shape& shape) = 0;

  virtual std::shared_ptr<runtime::Tensor> create_batched_plain_tensor(
      const element::Type& element_type, const Shape& shape) = 0;

  std::shared_ptr<runtime::Tensor> create_tensor(
      const element::Type& element_type, const Shape& shape,
      void* memory_pointer) override;

  std::shared_ptr<runtime::Tensor> create_plain_tensor(
      const element::Type& element_type, const Shape& shape,
      const bool batched = false);

  std::shared_ptr<runtime::Tensor> create_cipher_tensor(
      const element::Type& element_type, const Shape& shape,
      const bool batched = false);

  /// @brief Creates ciphertext Tensor of the same value
  /// @param value Scalar which to enrypt
  /// @param element_type Type to encrypt
  /// @param shape Shape of created Tensor
  std::shared_ptr<runtime::Tensor> create_valued_cipher_tensor(
      float value, const element::Type& element_type, const Shape& shape);

  // Creates plaintext Tensor of the same value
  /// @param value Scalar which to encode
  /// @param element_type Type to encode
  /// @param shape Shape of created Tensor
  std::shared_ptr<runtime::Tensor> create_valued_plain_tensor(
      float value, const element::Type& element_type, const Shape& shape);

  std::shared_ptr<Executable> compile(
      std::shared_ptr<Function> func,
      bool enable_performance_data = false) override;

  void validate_he_call(
      std::shared_ptr<const Function> function,
      const std::vector<std::shared_ptr<runtime::he::HETensor>>& outputs,
      const std::vector<std::shared_ptr<runtime::he::HETensor>>& inputs);

  /// @brief Encodes bytes to a plaintext polynomial
  /// @param output Pointer to plaintext to write to
  /// @param input Pointer to memory to encode
  /// @param type Type of scalar to encode
  /// @param count Number of elements to encode, count > 1 indicates batching
  virtual void encode(std::shared_ptr<runtime::he::HEPlaintext>& output,
                      const void* input, const element::Type& element_type,
                      size_t count = 1) const = 0;

  /// @brief Decodes plaintext polynomial to bytes
  /// @param output Pointer to memory to write to
  /// @param input Pointer to plaintext to decode
  /// @param type Type of scalar to encode
  /// @param count Number of elements to decode, count > 1 indicates batching
  virtual void decode(void* output, const runtime::he::HEPlaintext* input,
                      const element::Type& element_type,
                      size_t count = 1) const = 0;

  /// @brief Encrypts plaintext polynomial to ciphertext
  /// @param output Pointer to ciphertext to encrypt to
  /// @param input Pointer to plaintext to encrypt
  virtual void encrypt(std::shared_ptr<runtime::he::HECiphertext>& output,
                       runtime::he::HEPlaintext* input) = 0;

  /// @brief Decrypts ciphertext to plaintext polynomial
  /// @param output Pointer to plaintext to decrypt to
  /// @param input Pointer to ciphertext to decrypt
  virtual void decrypt(std::shared_ptr<runtime::he::HEPlaintext>& output,
                       const runtime::he::HECiphertext* input) const = 0;

  void set_batch_data(bool batch) { m_batch_data = batch; };

  const std::shared_ptr<HEEncryptionParameters> get_encryption_parameters()
      const {
    return m_encryption_params;
  };

  bool encrypt_data() const { return m_encrypt_data; };
  bool batch_data() const { return m_batch_data; };
  bool encrypt_model() const { return m_encrypt_model; };

 protected:
  bool m_encrypt_data{std::getenv("NGRAPH_ENCRYPT_DATA") != nullptr};
  bool m_batch_data{std::getenv("NGRAPH_BATCH_DATA") != nullptr};
  bool m_encrypt_model{std::getenv("NGRAPH_ENCRYPT_MODEL") != nullptr};

  std::shared_ptr<HEEncryptionParameters> m_encryption_params;
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
