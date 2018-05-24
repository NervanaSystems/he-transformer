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
        class CallFrame;

        namespace he
        {
            class HECallFrame;
            class HETensorView;
            class HEPlainTensorView;
            class HECipherTensorView;
            class HEBackend;

            namespace he_heaan
            {
                class HEHeaanBackend : public HEBackend,
                                       public std::enable_shared_from_this<HEHeaanBackend>
                {
                public:
                    HEHeaanBackend();
                    HEHeaanBackend(const std::shared_ptr<runtime::he::HEParameter> hep);
                    HEHeaanBackend(const std::shared_ptr<runtime::he::HEHeaanParameter> sp);
                    HEHeaanBackend(HEHeaanBackend& he_backend) = default;
                    ~HEHeaanBackend();

                    std::shared_ptr<heaan::Context>
                        make_heaan_context(const std::shared_ptr<runtime::he::HEHeaanParameter> sp);

                    std::shared_ptr<runtime::TensorView>
                        create_tensor(const element::Type& element_type,
                                      const Shape& shape) override;

                    std::shared_ptr<runtime::TensorView>
                        create_tensor(const element::Type& element_type,
                                      const Shape& shape,
                                      void* memory_pointer) override;

                    std::shared_ptr<runtime::TensorView>
                        create_plain_tensor(const element::Type& element_type, const Shape& shape);

                    // Create scalar text without memory pool
                    std::shared_ptr<he::HECiphertext>
                        create_valued_ciphertext(float value,
                                                 const element::Type& element_type) const;
                    std::shared_ptr<he::HECiphertext> create_empty_ciphertext() const;
                    std::shared_ptr<he::HEPlaintext>
                        create_valued_plaintext(float value,
                                                const element::Type& element_type) const;
                    std::shared_ptr<he::HEPlaintext> create_empty_plaintext() const;

                    // Create TensorView of the same value
                    std::shared_ptr<runtime::TensorView> create_valued_tensor(
                        float value, const element::Type& element_type, const Shape& shape);
                    std::shared_ptr<runtime::TensorView> create_valued_plain_tensor(
                        float value, const element::Type& element_type, const Shape& shape);

                    bool compile(std::shared_ptr<Function> func) override;

                    bool call(
                        std::shared_ptr<Function> func,
                        const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
                        const std::vector<std::shared_ptr<runtime::TensorView>>& inputs) override;

                    void clear_function_instance();

                    void remove_compiled_function(std::shared_ptr<Function> func) override;

                    void encode(std::shared_ptr<he::HEPlaintext>& output,
                                const void* input,
                                const element::Type& type);

                    void decode(void* output,
                                const std::shared_ptr<he::HEPlaintext> input,
                                const element::Type& type);

                    void encrypt(std::shared_ptr<he::HECiphertext> output,
                                 const std::shared_ptr<he::HEPlaintext> input);

                    void decrypt(std::shared_ptr<he::HEPlaintext> output,
                                 const std::shared_ptr<he::HECiphertext> input);

                    const inline std::shared_ptr<heaan::Scheme> get_scheme() const
                    {
                        return m_scheme;
                    }

                    void enable_performance_data(std::shared_ptr<Function> func,
                                                 bool enable) override;
                    std::vector<PerformanceCounter>
                        get_performance_data(std::shared_ptr<Function> func) const override;

                    void visualize_function_after_pass(const std::shared_ptr<Function>& func,
                                                       const std::string& file_name);

                private:
                    std::unordered_map<std::shared_ptr<Function>, std::shared_ptr<HECallFrame>>
                        m_function_map;
                    std::shared_ptr<heaan::SecretKey> m_secret_key;

                    std::shared_ptr<heaan::Context> m_context;
                    std::shared_ptr<heaan::Scheme> m_scheme;

                    long m_log_precision; // Bits of precision
                };
            }
        }
    }
}
