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
#include "seal/he_seal_backend.hpp"
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
                class HESealCKKSBackend : public HESealBackend
                {
                public:
                    HESealCKKSBackend();
                    HESealCKKSBackend(const std::shared_ptr<runtime::he::he_seal::HESealParameter>& sp);
                    HESealCKKSBackend(HESealCKKSBackend& he_backend) = default;
                    ~HESealCKKSBackend() {};

                     std::shared_ptr<runtime::Tensor> create_batched_tensor(
                        const element::Type& element_type, const Shape& shape) override;

                    void encode(std::shared_ptr<runtime::he::HEPlaintext>& output,
                                const void* input,
                                const element::Type& type,
                                size_t count = 1) const override;
                    void decode(void* output,
                                const std::shared_ptr<runtime::he::HEPlaintext> input,
                                const element::Type& type,
                                size_t count = 1) const override;

                    const inline std::shared_ptr<seal::CKKSEncoder> get_ckks_encoder() const
                    {
                        return m_ckks_encoder;
                    }

                    void assert_valid_seal_ckks_parameter(const std::shared_ptr<runtime::he::he_seal::HESealParameter>& sp) const;

                private:
                    std::shared_ptr<seal::CKKSEncoder> m_ckks_encoder;
                    double m_scale;
                };
            }
        }
    }
}
