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
#include "he_heaan_backend.hpp"
#include "he_heaan_parameter.hpp"
#include "heaan/heaan.hpp"
#include "heaan_ciphertext_wrapper.hpp"
#include "heaan_plaintext_wrapper.hpp"
#include "ngraph/runtime/backend.hpp"

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

            namespace he_heaan
            {
                class HEHeaanBackend : public HEBackend
                {
                public:
                    HEHeaanBackend();
                    HEHeaanBackend(const std::shared_ptr<runtime::he::HEHeaanParameter> sp);
                    HEHeaanBackend(HEHeaanBackend& he_backend) = default;
                    ~HEHeaanBackend();

                    /// @brief Checks if parameter is valid for HEAAN encoding.
                    ///        Throws an error if parameter is not valid.
                    void assert_valid_heaan_parameter(
                        const std::shared_ptr<runtime::he::HEHeaanParameter> hp) const;

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
                        std::shared_ptr<runtime::he::HECiphertext>& output,
                        const std::shared_ptr<runtime::he::HEPlaintext> input) const override;

                    void decrypt(
                        std::shared_ptr<runtime::he::HEPlaintext>& output,
                        const std::shared_ptr<runtime::he::HECiphertext> input) const override;

                    const inline std::shared_ptr<heaan::Scheme> get_scheme() const
                    {
                        return m_scheme;
                    }

                    const inline std::shared_ptr<heaan::Context> get_context() const
                    {
                        return m_context;
                    }

                    const inline long get_precision() const { return m_log2_precision; }
                    const inline std::shared_ptr<heaan::SecretKey> get_secret_key() const
                    {
                        return m_secret_key;
                    }



                private:
                    std::shared_ptr<heaan::SecretKey> m_secret_key;
                    std::shared_ptr<heaan::Context> m_context;
                    std::shared_ptr<heaan::Scheme> m_scheme;

                    long m_log2_precision; // Bits of precision
                };
            }
        }
    }
}
