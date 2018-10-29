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
#include <functional>
#include <memory>
#include <vector>

#include "he_tensor.hpp"

#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/performance_counter.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/node.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            class HECallFrame;
            class HETensor;
            class HEPlainTensor;
            class HECipherTensor;
            class HECiphertext;
            class HEPlaintext;

            class HEBackend : public runtime::Backend
            {
            public:
                HEBackend() {};
                HEBackend(HEBackend& he_backend) = default;
                ~HEBackend() {};

                std::shared_ptr<runtime::Tensor>
                    create_tensor(const element::Type& element_type,
                                  const Shape& shape) override;

                virtual std::shared_ptr<runtime::Tensor> create_batched_tensor(
                    const element::Type& element_type, const Shape& shape) = 0;

                /// @brief Return a handle for a tensor for given mem on backend device
                std::shared_ptr<runtime::Tensor>
                    create_tensor(const element::Type& element_type,
                                  const Shape& shape,
                                  void* memory_pointer) override;

                std::shared_ptr<runtime::Tensor>
                    create_plain_tensor(const element::Type& element_type, const Shape& shape);

                /// @brief Creates ciphertext of specified value
                /// @param value Scalar which to encrypt
                /// @param element_type Type to encrypt
                /// @param batch_size Number of elements to encrypt
                ///        > 1 indicates batching
                /// @return Shared pointer to created ciphertext
                std::shared_ptr<runtime::he::HECiphertext>
                    create_valued_ciphertext(float value,
                                             const element::Type& element_type,
                                             size_t batch_size = 1) const;

                /// @brief Creates ciphertext of unspecified value
                /// @return Shared pointer to created ciphertext
                virtual std::shared_ptr<runtime::he::HECiphertext>
                    create_empty_ciphertext() const = 0;

                /// @brief Creates plaintext of specified value
                /// @param value Scalar which to encode
                /// @param element_type Type to encode
                /// @return Shared pointer to created plaintext
                std::shared_ptr<runtime::he::HEPlaintext>
                    create_valued_plaintext(float value,
                                            const element::Type& element_type) const;

                /// @brief Creates plaintext of unspecified value
                /// @return Shared pointer to created plaintext
                virtual std::shared_ptr<runtime::he::HEPlaintext>
                    create_empty_plaintext() const = 0;

                /// @brief Creates ciphertext Tensor of the same value
                /// @param value Scalar which to enrypt
                /// @param element_type Type to encrypt
                /// @param shape Shape of created Tensor
                std::shared_ptr<runtime::Tensor> create_valued_tensor(
                    float value, const element::Type& element_type, const Shape& shape);

                // Creates plaintext Tensor of the same value
                /// @param value Scalar which to encode
                /// @param element_type Type to encode
                /// @param shape Shape of created Tensor
                std::shared_ptr<runtime::Tensor> create_valued_plain_tensor(
                    float value, const element::Type& element_type, const Shape& shape);

                bool compile(std::shared_ptr<Function> function) override;

                bool call(std::shared_ptr<Function> function,
                          const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                          const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) override;

                void clear_function_instance();

                void remove_compiled_function(std::shared_ptr<Function> function) override;

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
                virtual void
                    encrypt(std::shared_ptr<runtime::he::HECiphertext>& output,
                            const std::shared_ptr<runtime::he::HEPlaintext> input) const = 0;

                /// @brief Decrypts ciphertext to plaintext polynomial
                /// @param output Pointer to plaintext to decrypt to
                /// @param input Pointer to ciphertext to decrypt
                virtual void
                    decrypt(std::shared_ptr<runtime::he::HEPlaintext>& output,
                            const std::shared_ptr<runtime::he::HECiphertext> input) const = 0;

                void enable_performance_data(std::shared_ptr<Function> function, bool enable) override;
                std::vector<PerformanceCounter>
                    get_performance_data(std::shared_ptr<Function> function) const override;

            private:
                class FunctionInstance
                {
                public:
                    bool m_is_compiled = false;
                    bool m_nan_check_enabled = false;
                    bool m_performance_counters_enabled = false;
                    std::unordered_map<const Node*, stopwatch> m_timer_map;
                    std::vector<std::shared_ptr<Node>> m_nodes;
                };
                std::map<std::shared_ptr<Function>, FunctionInstance> m_function_map;

                void generate_calls(const element::Type& type,
                    const std::shared_ptr<Node>& op,
                    const std::vector<std::shared_ptr<runtime::he::HETensor>>& outputs,
                    const std::vector<std::shared_ptr<runtime::he::HETensor>>& inputs);

                bool call(std::shared_ptr<Function> function,
                    const std::vector<std::shared_ptr<runtime::he::HETensor>>& outputs,
                    const std::vector<std::shared_ptr<runtime::he::HETensor>>& inputs);

            protected:
                std::unordered_map<
                    std::string,
                    std::unordered_map<std::int64_t, std::shared_ptr<runtime::he::HEPlaintext>>>
                    m_plaintext_map;
            };
        }
    }
}
