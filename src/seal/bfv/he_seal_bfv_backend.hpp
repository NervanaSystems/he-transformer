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

#include "seal/he_seal_parameter.hpp"
#include "ngraph/runtime/backend.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            class HETensor;
            class HEPlainTensor;
            class HECipherTensor;
            class HESealBackend;

            namespace he_seal
            {
                class HESealBFVBackend : public HESealBackend
                {
                public:
                    HESealBFVBackend();
                    HESealBFVBackend(const std::shared_ptr<runtime::he::he_seal::HESealParameter>& sp);
                    HESealBFVBackend(HESealBFVBackend& he_backend) = default;
                    ~HESealBFVBackend();

                    std::shared_ptr<runtime::Tensor> create_batched_tensor(const element::Type& element_type,
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
                    /* std::shared_ptr<runtime::he::HECiphertext>
                        create_valued_ciphertext(float value,
                                                 const element::Type& element_type,
                                                 size_t batch_size = 1) const override; */

                   /*  std::shared_ptr<runtime::he::HEPlaintext>
                        create_valued_plaintext(float value,
                                                const element::Type& element_type) const override;

                    std::shared_ptr<runtime::he::HEPlaintext>
                        get_valued_plaintext(std::int64_t value,
                                             const element::Type& element_type) override; */

                    // Create Tensor of the same value
                   /*  std::shared_ptr<runtime::Tensor>
                        create_valued_tensor(float value,
                                             const element::Type& element_type,
                                             const Shape& shape) override;
                    std::shared_ptr<runtime::Tensor>
                        create_valued_plain_tensor(float value,
                                                   const element::Type& element_type,
                                                   const Shape& shape) override; */

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

                    const inline std::shared_ptr<seal::IntegerEncoder> get_int_encoder() const
                    {
                        return m_int_encoder;
                    }

                    const inline std::shared_ptr<seal::FractionalEncoder> get_frac_encoder() const
                    {
                        return m_frac_encoder;
                    }

                    /* struct plaintext_num
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
                    } */

                    /// @brief Checks the noise budget of several tensor views
                    ///        Throws an error if the noise budget is exhauasted
                    ///        for any of the tensor views.
                    void check_noise_budget(
                        const std::vector<std::shared_ptr<runtime::he::HETensor>>& tvs) const;

                    /// @brief Returns the remaining noise budget for a ciphertext.
                    //         A noise budget of <= 0 indicate the ciphertext is no longer
                    //         decryptable.
                    int noise_budget(const std::shared_ptr<seal::Ciphertext>& ciphertext) const;

                private:
                    std::shared_ptr<seal::IntegerEncoder> m_int_encoder;
                    std::shared_ptr<seal::FractionalEncoder> m_frac_encoder;
                };
            }
        }
    }
}
