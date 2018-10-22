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
#include "seal/he_seal_parameter.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"
#include "ngraph/runtime/backend.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            class HETensor;
            class HEPlainTensor;
            class HECipherTensor;
            class HEBackend;

            namespace he_seal
            {
                class HESealBackend : public HEBackend
                {
                public:
                    HESealBackend();
                    HESealBackend(const std::shared_ptr<runtime::he::he_seal::HESealParameter>& sp);

                    /// @brief Checks if parameter is valid for HEAAN encoding.
                    ///        Throws an error if parameter is not valid.
                    virtual void assert_valid__parameter(
                        const std::shared_ptr<runtime::he::he_seal::HESealParameter> hp) const = 0;

                    std::shared_ptr<runtime::Tensor>
                        create_tensor(const element::Type& element_type,
                                      const Shape& shape) override;

                    std::shared_ptr<runtime::Tensor>
                        create_tensor(const element::Type& element_type,
                                      const Shape& shape,
                                      const bool batched) override;

                    std::shared_ptr<runtime::Tensor>
                        create_tensor(const element::Type& element_type,
                                      const Shape& shape,
                                      void* memory_pointer) override;

                    std::shared_ptr<runtime::Tensor>
                        create_plain_tensor(const element::Type& element_type,
                                            const Shape& shape) override;

                    std::shared_ptr<runtime::he::HECiphertext>&
                        get_valued_ciphertext(std::int64_t value,
                                              const element::Type& element_type,
                                              size_t batch_size = 1);

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
                                             const element::Type& element_type) override;

                    std::shared_ptr<runtime::he::HEPlaintext>
                        create_empty_plaintext() const override;

                    std::shared_ptr<runtime::Tensor>
                        create_valued_tensor(float value,
                                             const element::Type& element_type,
                                             const Shape& shape) override;

                    std::shared_ptr<runtime::Tensor>
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
                        std::shared_ptr<runtime::he::HECiphertext>& output,
                        const std::shared_ptr<runtime::he::HEPlaintext> input) const override;

                    void decrypt(
                        std::shared_ptr<runtime::he::HEPlaintext>& output,
                        const std::shared_ptr<runtime::he::HECiphertext> input) const override;

                    const inline std::shared_ptr<seal::SEALContext> get_context() const
                    {
                        return m_context;
                    }

                    const inline std::shared_ptr<seal::SecretKey> get_secret_key() const
                    {
                        return m_secret_key;
                    }

                private:
                    std::shared_ptr<seal::SecretKey> m_secret_key;
                    std::shared_ptr<seal::SEALContext> m_context;
                };
            }
        }
    }
}
