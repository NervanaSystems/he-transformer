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
#include "he_parameter.hpp"
#include "he_ciphertext.hpp"
#include "he_plaintext.hpp"

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

            class HEBackend : public runtime::Backend,
                              public std::enable_shared_from_this<HEBackend>
            {
            public:
                HEBackend();
                HEBackend(const std::shared_ptr<runtime::he::HEParameter> hep);
                HEBackend(HEBackend& he_backend) = default;
                // virtual ~HEBackend();

                std::shared_ptr<runtime::TensorView>
                    create_tensor(const element::Type& element_type, const Shape& shape) override;

                std::shared_ptr<runtime::TensorView>
                    create_tensor(const element::Type& element_type,
                                  const Shape& shape,
                                  void* memory_pointer) override;

                std::shared_ptr<runtime::TensorView>
                    create_plain_tensor(const element::Type& element_type, const Shape& shape);

                // Create scalar text with memory pool
                std::shared_ptr<he::HECiphertext>
                    create_valued_ciphertext(float value,
                                             const element::Type& element_type) const;
                std::shared_ptr<he::HECiphertext>
                    create_empty_ciphertext() const;
                std::shared_ptr<he::HEPlaintext>
                    create_valued_plaintext(float value,
                                            const element::Type& element_type) const;
                std::shared_ptr<he::HEPlaintext>
                    create_empty_plaintext() const;

                // Create TensorView of the same value
                std::shared_ptr<runtime::TensorView> create_valued_tensor(
                    float value, const element::Type& element_type, const Shape& shape);
                std::shared_ptr<runtime::TensorView> create_valued_plain_tensor(
                    float value, const element::Type& element_type, const Shape& shape);

                bool compile(std::shared_ptr<Function> func) override;

                bool call(std::shared_ptr<Function> func,
                          const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
                          const std::vector<std::shared_ptr<runtime::TensorView>>& inputs) override;

                void clear_function_instance();

                void remove_compiled_function(std::shared_ptr<Function> func) override;

                void encode(he::HEPlaintext& output, const void* input, const element::Type& type);

                void decode(void* output, const he::HEPlaintext& input, const element::Type& type);

                void encrypt(he::HECiphertext& output, const he::HEPlaintext& input);

                void decrypt(he::HEPlaintext& output, const he::HECiphertext& input);

                // void check_noise_budget(const std::vector<std::shared_ptr<runtime::he::HETensorView>>& tvs) const;

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
