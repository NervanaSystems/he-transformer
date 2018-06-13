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

#include "he_backend.hpp"
#include "he_seal_parameter.hpp"
#include "ngraph/runtime/backend.hpp"
#include "seal/seal.h"
#include "seal_ciphertext_wrapper.hpp"
#include "seal_plaintext_wrapper.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            class HETensorView;
            class HEPlainTensorView;
            class HECipherTensorView;
            class HEBackend;

            namespace he_seal
            {
                class HESealBackend : public HEBackend
                {
                public:
                    HESealBackend();
                    HESealBackend(const std::shared_ptr<runtime::he::HESealParameter> sp);
                    HESealBackend(HESealBackend& he_backend) = default;
                    ~HESealBackend();

                    /// @brief Checks if parameter is valid for SEAL encoding.
                    //         Throws an error if parameter is not valid.
                    void assert_valid_seal_parameter(
                        const std::shared_ptr<runtime::he::HESealParameter> sp) const;

                    /// @brief Constructs SEAL context from SEAL parameter
                    /// @param sp SEAL Parameter from which to construct context
                    /// @return Pointer to constructed context
                    shared_ptr<seal::SEALContext> make_seal_context(
                        const std::shared_ptr<runtime::he::HESealParameter> sp) const;

                    std::shared_ptr<runtime::TensorView>
                        create_tensor(const element::Type& element_type,
                                      const Shape& shape) override;

                    std::shared_ptr<runtime::TensorView>
                        create_tensor(const element::Type& element_type,
                                      const Shape& shape,
                                      const bool batched) override;

                    std::shared_ptr<runtime::TensorView>
                        create_tensor(const element::Type& element_type,
                                      const Shape& shape,
                                      void* memory_pointer) override;

                    std::shared_ptr<runtime::TensorView>
                        create_plain_tensor(const element::Type& element_type,
                                            const Shape& shape) override;

                    // Create scalar text with memory pool
                    std::shared_ptr<runtime::he::HECiphertext>
                        create_valued_ciphertext(float value,
                                                 const element::Type& element_type,
                                                 const seal::MemoryPoolHandle& pool) const;
                    std::shared_ptr<runtime::he::HECiphertext>
                        create_empty_ciphertext(const seal::MemoryPoolHandle& pool) const;

                    std::shared_ptr<runtime::he::HEPlaintext>
                        create_valued_plaintext(float value,
                                                const element::Type& element_type,
                                                const seal::MemoryPoolHandle& pool) const;

                    std::shared_ptr<runtime::he::HEPlaintext>
                        create_empty_plaintext(const seal::MemoryPoolHandle& pool) const;

                    // Create scalar text without memory pool
                    std::shared_ptr<runtime::he::HECiphertext>
                        create_valued_ciphertext(float value,
                                                 const element::Type& element_type,
                                                 size_t batch_size = 1) const override;

                    std::shared_ptr<runtime::he::HECiphertext>
                        create_empty_ciphertext(size_t batch_size = 1) const override;

                    std::shared_ptr<runtime::he::HEPlaintext>
                        create_valued_plaintext(float value,
                                                const element::Type& element_type) const override;

                    std::shared_ptr<runtime::he::HEPlaintext>
                        get_valued_plaintext(std::int64_t value,
                                const element::Type& element_type)override;

                    std::shared_ptr<runtime::he::HEPlaintext>
                        create_empty_plaintext() const override;

                    // Create TensorView of the same value
                    std::shared_ptr<runtime::TensorView>
                        create_valued_tensor(float value,
                                             const element::Type& element_type,
                                             const Shape& shape) override;
                    std::shared_ptr<runtime::TensorView>
                        create_valued_plain_tensor(float value,
                                                   const element::Type& element_type,
                                                   const Shape& shape) override;

                    void encode(std::shared_ptr<runtime::he::HEPlaintext>& output,
                                const void* input,
                                const element::Type& type,
                                size_t count = 1) const override;
                    void decode(void* output,
                                const std::shared_ptr<runtime::he::HEPlaintext> input,
                                const element::Type& type,
                                size_t count = 1) const override;

                    void encrypt(
                        shared_ptr<runtime::he::HECiphertext>& output,
                        const std::shared_ptr<runtime::he::HEPlaintext> input) const override;

                    void decrypt(
                        std::shared_ptr<runtime::he::HEPlaintext>& output,
                        const std::shared_ptr<runtime::he::HECiphertext> input) const override;

                    const inline std::shared_ptr<seal::SEALContext> get_context() const
                    {
                        return m_context;
                    }

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
                        seal::Plaintext fl_0;
                        seal::Plaintext fl_1;
                        seal::Plaintext fl_n1;
                        seal::Plaintext int64_0;
                        seal::Plaintext int64_1;
                        seal::Plaintext int64_n1;
                    };

                    const inline plaintext_num& get_plaintext_num() const
                    {
                        return m_plaintext_num;
                    }

                    /// @brief Checks the noise budget of several tensor views
                    ///        Throws an error if the noise budget is exhauasted
                    ///        for any of the tensor views.
                    void check_noise_budget(
                        const vector<shared_ptr<runtime::he::HETensorView>>& tvs) const;

                    /// @brief Returns the remaining noise budget for a ciphertext.
                    //         A noise budget of <= 0 indicate the ciphertext is no longer
                    //         decryptable.
                    int noise_budget(const std::shared_ptr<seal::Ciphertext>& ciphertext) const;

                private:
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
                };
            }
        }
    }
}
