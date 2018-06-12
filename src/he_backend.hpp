/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <memory>
#include <unordered_map>

#include "he_parameter.hpp"
#include "ngraph/runtime/backend.hpp"

namespace ngraph
{
    namespace runtime
    {
        class CallFrame;

        namespace he
        {
            class HECallFrame;
            class HETensorView;
            class HEPlainTensorView;
            class HECipherTensorView;
            class HECiphertext;
            class HEPlaintext;

            class HEBackend : public runtime::Backend
            {
            public:
                HEBackend();
                HEBackend(const std::shared_ptr<runtime::he::HEParameter> hep);
                HEBackend(HEBackend& he_backend) = default;

                virtual std::shared_ptr<runtime::TensorView>
                    create_tensor(const element::Type& element_type, const Shape& shape) override = 0;

                virtual std::shared_ptr<runtime::TensorView> create_tensor(
                    const element::Type& element_type, const Shape& shape, const bool batched) = 0;

                /// @brief Return a handle for a tensor for given mem on backend device
                std::shared_ptr<runtime::TensorView>
                    create_tensor(const element::Type& element_type,
                                  const Shape& shape,
                                  void* memory_pointer) override;

                virtual std::shared_ptr<runtime::TensorView>
                    create_plain_tensor(const element::Type& element_type, const Shape& shape) = 0;

                /// @brief Creates ciphertext of given value
                /// @param value Scalar which to encrypt
                /// @param element_type Type to encrypt
                /// @param batch_size Number of elements to encrypt
                ///        > 1 indicates batching
                /// @return Shared pointer to created ciphertext
                virtual std::shared_ptr<runtime::he::HECiphertext>
                    create_valued_ciphertext(float value, const element::Type& element_type, size_t batch_size = 1) const = 0;

                /// @brief Creates ciphertextof unspecified value
                /// @param batch_size Number of elements to encrypt in a
                ///        > 1 indicates batching
                /// @return Shared pointer to created ciphertext
                virtual std::shared_ptr<runtime::he::HECiphertext> create_empty_ciphertext(size_t batch_size = 1) const = 0;

                /// @brief Creates plaintext of unspecified value
                /// @param value Scalar which to encode
                /// @param element_type Type to encode
                /// @return Shared pointer to created plaintext
                virtual std::shared_ptr<runtime::he::HEPlaintext>
                    create_valued_plaintext(float value, const element::Type& element_type) const = 0;

                /// @brief Creates plaintext of unspecified value
                /// @return Shared pointer to created plaintext
                virtual std::shared_ptr<runtime::he::HEPlaintext> create_empty_plaintext() const = 0;

                /// @brief Creates ciphertext TensorView of the same value
                /// @param value Scalar which to enrypt
                /// @param element_type Type to encrypt
                /// @param shape Shape of created TensorView
                virtual std::shared_ptr<runtime::TensorView> create_valued_tensor(
                    float value, const element::Type& element_type, const Shape& shape) = 0;

                // Creates plaintext TensorView of the same value
                /// @param value Scalar which to encode
                /// @param element_type Type to encode
                /// @param shape Shape of created TensorView
                virtual std::shared_ptr<runtime::TensorView> create_valued_plain_tensor(
                    float value, const element::Type& element_type, const Shape& shape) = 0;

                bool compile(std::shared_ptr<Function> func) override;

                bool call(std::shared_ptr<Function> func,
                          const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
                          const std::vector<std::shared_ptr<runtime::TensorView>>& inputs) override;

                void clear_function_instance();

                void remove_compiled_function(std::shared_ptr<Function> func) override;

                /// @brief Encodes bytes to a plaintext polynomial
                /// @param output Pointer to plaintext to write to
                /// @param input Pointer to memory to encode
                /// @param type Type of scalar to encode
                /// @param count Number of elements to encode, count > 1 indicates batching
                virtual void encode(std::shared_ptr<runtime::he::HEPlaintext>& output,
                            const void* input,
                            const element::Type& type,
                            size_t count = 1) const = 0;

                /// @brief Decodes plaintext polynomial to bytes
                /// @param output Pointer to memory to write to
                /// @param input Pointer to plaintext to decode
                /// @param type Type of scalar to encode
                /// @param count Number of elements to decode, count > 1 indicates batching
                virtual void decode(void* output,
                            const std::shared_ptr<runtime::he::HEPlaintext> input,
                            const element::Type& type,
                            size_t count = 1) const = 0;

                /// @brief Encrypts plaintext polynomial to ciphertext
                /// @param output Pointer to ciphertext to encrypt to
                /// @param input Pointer to plaintext to encrypt
                virtual void encrypt(std::shared_ptr<runtime::he::HECiphertext>& output,
                             const std::shared_ptr<runtime::he::HEPlaintext> input) const = 0;

                /// @brief Decrypts ciphertext to plaintext polynomial
                /// @param output Pointer to plaintext to decrypt to
                /// @param input Pointer to ciphertext to decrypt
                virtual void decrypt(std::shared_ptr<runtime::he::HEPlaintext>& output, const std::shared_ptr<runtime::he::HECiphertext> input) const = 0;

                void enable_performance_data(std::shared_ptr<Function> func, bool enable) override;
                std::vector<PerformanceCounter>
                    get_performance_data(std::shared_ptr<Function> func) const override;

                void visualize_function_after_pass(const std::shared_ptr<Function>& func,
                                                   const std::string& file_name);

            private:
                std::unordered_map<std::shared_ptr<Function>, std::shared_ptr<HECallFrame>>
                    m_function_map;
            };
        }
    }
}
