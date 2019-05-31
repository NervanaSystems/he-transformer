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
namespace he {
class HETensor;
class HEBackend : public ngraph::runtime::Backend {
 public:
  virtual ~HEBackend(){};

  /// @brief Creates ciphertext of unspecified value
  /// @return Shared pointer to created ciphertext
  virtual std::shared_ptr<ngraph::he::SealCiphertextWrapper> create_empty_ciphertext()
      const = 0;

  /// @brief Creates ciphertext of unspecified value
  /// Alias for create_empty_ciphertext()
  /// @return Shared pointer to created ciphertext
  template <
      typename T,
      typename = std::enable_if_t<
          std::is_same<T, ngraph::he::SealCiphertextWrapper>::value ||
          std::is_same<T, std::shared_ptr<ngraph::he::SealCiphertextWrapper>>::value>>
  std::shared_ptr<ngraph::he::SealCiphertextWrapper> create_empty_hetext() const {
    return create_empty_ciphertext();
  };

  /// @brief Creates ciphertext of specified value
  /// @param value Scalar which to encrypt
  /// @param element_type Type to encrypt
  /// @param batch_size Number of elements to encrypt
  ///        > 1 indicates batching
  /// @return Shared pointer to created ciphertext
  std::shared_ptr<ngraph::he::SealCiphertextWrapper> create_valued_ciphertext(
      float value, const element::Type& element_type,
      size_t batch_size = 1) const;

  /// @brief Creates ciphertext of specified value
  /// Alias for create_valued_ciphertext()
  /// @return Shared pointer to created ciphertext
  template <typename T, typename = std::enable_if_t<
                            std::is_same<T, ngraph::he::SealCiphertextWrapper>::value>>
  std::shared_ptr<ngraph::he::SealCiphertextWrapper> create_valued_hetext(
      float value, const element::Type& element_type,
      size_t batch_size = 1) const {
    return create_valued_ciphertext(value, element_type, batch_size);
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
      const bool batched = false) const;

  std::shared_ptr<runtime::Tensor> create_cipher_tensor(
      const element::Type& element_type, const Shape& shape,
      const bool batched = false) const;

  /// @brief Creates ciphertext Tensor of the same value
  /// @param value Scalar which to enrypt
  /// @param element_type Type to encrypt
  /// @param shape Shape of created Tensor
  std::shared_ptr<runtime::Tensor> create_valued_cipher_tensor(
      float value, const element::Type& element_type, const Shape& shape) const;

  // Creates plaintext Tensor of the same value
  /// @param value Scalar which to encode
  /// @param element_type Type to encode
  /// @param shape Shape of created Tensor
  std::shared_ptr<runtime::Tensor> create_valued_plain_tensor(
      float value, const element::Type& element_type, const Shape& shape) const;

  std::shared_ptr<ngraph::runtime::Executable> compile(
      std::shared_ptr<Function> func,
      bool enable_performance_data = false) override;

  void validate_he_call(
      std::shared_ptr<const Function> function,
      const std::vector<std::shared_ptr<ngraph::he::HETensor>>& outputs,
      const std::vector<std::shared_ptr<ngraph::he::HETensor>>& inputs);

  /// @brief Encodes bytes to a plaintext polynomial
  /// @param output Pointer to plaintext to write to
  /// @param input Pointer to memory to encode
  /// @param type Type of scalar to encode
  /// @param count Number of elements to encode, count > 1 indicates batching
  virtual void encode(ngraph::he::HEPlaintext& output, const void* input,
                      const element::Type& element_type, bool complex = false,
                      size_t count = 1) const = 0;

  /// @brief Decodes plaintext polynomial to bytes
  /// @param output Pointer to memory to write to
  /// @param input Pointer to plaintext to decode
  /// @param type Type of scalar to encode
  /// @param count Number of elements to decode, count > 1 indicates batching
  virtual void decode(void* output, const ngraph::he::HEPlaintext& input,
                      const element::Type& element_type,
                      size_t count = 1) const = 0;

  /// @brief Encrypts plaintext polynomial to ciphertext
  /// @param output Pointer to ciphertext to encrypt to
  /// @param input Pointer to plaintext to encrypt
  virtual void encrypt(std::shared_ptr<ngraph::he::SealCiphertextWrapper>& output,
                       const ngraph::he::HEPlaintext& input) const = 0;

  /// @brief Decrypts ciphertext to plaintext polynomial
  /// @param output Pointer to plaintext to decrypt to
  /// @param input Pointer to ciphertext to decrypt
  virtual void decrypt(
      ngraph::he::HEPlaintext& output,
      const std::shared_ptr<ngraph::he::SealCiphertextWrapper>& input) const = 0;

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

 protected:
  bool m_encrypt_data{std::getenv("NGRAPH_ENCRYPT_DATA") != nullptr};
  bool m_batch_data{std::getenv("NGRAPH_BATCH_DATA") != nullptr};
  bool m_encrypt_model{std::getenv("NGRAPH_ENCRYPT_MODEL") != nullptr};
  bool m_complex_packing{std::getenv("NGRAPH_COMPLEX_PACK") != nullptr};

  std::shared_ptr<HEEncryptionParameters> m_encryption_params;
};
}  // namespace he
}  // namespace ngraph