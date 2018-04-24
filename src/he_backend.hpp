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

#include "ngraph/runtime/backend.hpp"
#include "seal/seal.h"

namespace ngraph
{
    namespace runtime
    {
        class CallFrame;

        namespace he
        {
            class HEExternalFunction;
            class HECallFrame;
            class HETensorView;
            class HEPlainTensorView;
            class HECipherTensorView;

            class HEBackend : public runtime::Backend,
                              public std::enable_shared_from_this<HEBackend>
            {
            public:
                HEBackend();
                HEBackend(seal::SEALContext& context);
                HEBackend(HEBackend& he_backend) = default;
                ~HEBackend();

                std::shared_ptr<runtime::TensorView>
                    create_tensor(const element::Type& element_type, const Shape& shape) override;
                std::shared_ptr<runtime::TensorView>
                    create_plain_tensor(const element::Type& element_type, const Shape& shape);

                std::shared_ptr<runtime::TensorView>
                    create_tensor(const element::Type& element_type,
                                  const Shape& shape,
                                  void* memory_pointer) override;
                std::shared_ptr<runtime::TensorView> create_plain_tensor(
                    const element::Type& element_type, const Shape& shape, void* memory_pointer);

                bool compile(std::shared_ptr<Function> func) override;

                bool call(std::shared_ptr<Function> func,
                          const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
                          const std::vector<std::shared_ptr<runtime::TensorView>>& inputs) override;

                void remove_compiled_function(std::shared_ptr<Function> func) override;

                void encode(seal::Plaintext& output, const void* input, const element::Type& type);
                void decode(void* output, const seal::Plaintext& input, const element::Type& type);

                void encrypt(seal::Ciphertext& output, const seal::Plaintext& input);

                void decrypt(seal::Plaintext& output, const seal::Ciphertext& input);

                const inline std::shared_ptr<seal::Evaluator> get_evaluator() const
                {
                    return m_evaluator;
                }

            private:
                seal::EncryptionParameters parms;
                std::shared_ptr<seal::SEALContext> m_context;
                std::shared_ptr<seal::IntegerEncoder> m_int_encoder;
                std::shared_ptr<seal::FractionalEncoder> m_frac_encoder;
                std::shared_ptr<seal::KeyGenerator> m_keygen;
                std::shared_ptr<seal::PublicKey> m_public_key;
                std::shared_ptr<seal::SecretKey> m_secret_key;
                std::shared_ptr<seal::Encryptor> m_encryptor;
                std::shared_ptr<seal::Decryptor> m_decryptor;
                std::shared_ptr<seal::Evaluator> m_evaluator;

                class FunctionInstance
                {
                public:
                    std::shared_ptr<HEExternalFunction> m_external_function;
                    std::shared_ptr<HECallFrame> m_call_frame;
                };

                std::unordered_map<std::shared_ptr<Function>, FunctionInstance> m_function_map;
            };
        }
    }
}
