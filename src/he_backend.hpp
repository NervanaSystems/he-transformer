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
#include "seal_parameter.hpp"

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

            class HEBackend : public runtime::Backend,
                              public std::enable_shared_from_this<HEBackend>
            {
            public:
                HEBackend();
                HEBackend(const runtime::he::SEALParameter& sp);
                HEBackend(HEBackend& he_backend) = default;
                ~HEBackend();

                std::shared_ptr<runtime::TensorView>
                    create_tensor(const element::Type& element_type, const Shape& shape) override;
                std::shared_ptr<runtime::TensorView>
                    create_plain_tensor(const element::Type& element_type, const Shape& shape);

                std::shared_ptr<runtime::TensorView>
                    create_zero_tensor(const element::Type& element_type, const Shape& shape);
                std::shared_ptr<runtime::TensorView>
                    create_ones_tensor(const element::Type& element_type, const Shape& shape);
                std::shared_ptr<runtime::TensorView> create_constant_tensor(
                    const element::Type& element_type, const Shape& shape, size_t element);

                std::shared_ptr<runtime::TensorView>
                    create_zero_plain_tensor(const element::Type& element_type, const Shape& shape);
                std::shared_ptr<runtime::TensorView>
                    create_ones_plain_tensor(const element::Type& element_type, const Shape& shape);
                std::shared_ptr<runtime::TensorView> create_constant_plain_tensor(
                    const element::Type& element_type, const Shape& shape, size_t element);

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

                void clear_function_instance();

                void remove_compiled_function(std::shared_ptr<Function> func) override;

                void encode(seal::Plaintext& output, const void* input, const element::Type& type);

                void decode(void* output, const seal::Plaintext& input, const element::Type& type);

                void encrypt(seal::Ciphertext& output, const seal::Plaintext& input);

                void decrypt(seal::Plaintext& output, const seal::Ciphertext& input);

                void check_noise_budget(
                    const vector<shared_ptr<runtime::he::HETensorView>>& tvs) const;

                const inline std::shared_ptr<seal::Evaluator> get_evaluator() const
                {
                    return m_evaluator;
                }

                const inline std::shared_ptr<seal::EvaluationKeys> get_ev_key() const
                {
                    return m_ev_key;
                }

                const inline std::shared_ptr<seal::Encryptor> get_encryptor() const
                {
                    return m_encryptor;
                }

                const inline std::shared_ptr<seal::Decryptor> get_decryptor() const
                {
                    return m_decryptor;
                }

                const inline std::shared_ptr<seal::IntegerEncoder> get_int_encoder() const
                {
                    return m_int_encoder;
                }

                const inline std::shared_ptr<seal::FractionalEncoder> get_frac_encoder() const
                {
                    return m_frac_encoder;
                }

                struct plaintext_num
                {
                    seal::Plaintext fl_1;
                    seal::Plaintext fl_n1;
                    seal::Plaintext int64_t_1;
                    seal::Plaintext int64_t_n1;
                };

                const inline plaintext_num& get_plaintext_num() const { return m_plaintext_num; }
                int noise_budget(const std::shared_ptr<seal::Ciphertext>& ciphertext) const;

                void enable_performance_data(std::shared_ptr<Function> func, bool enable) override;
                std::vector<PerformanceCounter>
                    get_performance_data(std::shared_ptr<Function> func) const override;

                void visualize_function_after_pass(const std::shared_ptr<Function>& func,
                                                   const std::string& file_name);

            private:
                seal::EncryptionParameters parms;
                std::shared_ptr<seal::SEALContext> m_context;
                std::shared_ptr<seal::IntegerEncoder> m_int_encoder;
                std::shared_ptr<seal::FractionalEncoder> m_frac_encoder;
                std::shared_ptr<seal::KeyGenerator> m_keygen;
                std::shared_ptr<seal::PublicKey> m_public_key;
                std::shared_ptr<seal::SecretKey> m_secret_key;
                std::shared_ptr<seal::EvaluationKeys> m_ev_key;
                std::shared_ptr<seal::Encryptor> m_encryptor;
                std::shared_ptr<seal::Decryptor> m_decryptor;
                std::shared_ptr<seal::Evaluator> m_evaluator;
                plaintext_num m_plaintext_num;
                std::unordered_map<std::shared_ptr<Function>, std::shared_ptr<HECallFrame>>
                    m_function_map;
            };
        }
    }
}
