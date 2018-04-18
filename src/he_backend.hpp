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

#include <map>
#include <memory>

#include "ngraph/function.hpp"
#include "ngraph/runtime/backend.hpp"
#include "seal/seal.h"

namespace ngraph
{
    namespace runtime
    {
        class CallFrame;

        namespace he
        {
            class ExternalFunction;
            class HECallFrame;
            class HETensorView;
            class HEPlainTensorView;
            class HECipherTensorView;

            class HEBackend : public runtime::Backend
            {
            public:
                HEBackend();
                HEBackend(seal::SEALContext& context);
                HEBackend(HEBackend& he_backend) = default;
                ~HEBackend();

                std::shared_ptr<runtime::TensorView>
                    create_tensor(const element::Type& element_type, const Shape& shape) override;

                std::shared_ptr<ngraph::runtime::TensorView>
                    create_tensor(const ngraph::element::Type& element_type,
                                  const Shape& shape,
                                  void* memory_pointer) override;

                bool compile(std::shared_ptr<Function> func) override;

                bool call(std::shared_ptr<Function> func,
                          const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
                          const std::vector<std::shared_ptr<runtime::TensorView>>& inputs) override;

                void remove_compiled_function(std::shared_ptr<Function> func) override;

                void encrypt(const HECipherTensorView& output, const HEPlainTensorView& input);

                void decrypt(const HEPlainTensorView& output, const HECipherTensorView& input);

                void encode(const HEPlainTensorView& output, const TensorView& input);

                void decode(const TensorView& output, const HEPlainTensorView& input);

            private:
                std::shared_ptr<seal::SEALContext> m_context;
            public: // TODO: make private and use encode function
                std::shared_ptr<seal::IntegerEncoder> m_int_encoder;
            private:
                std::shared_ptr<seal::KeyGenerator> m_keygen;
                std::shared_ptr<seal::PublicKey> m_public_key;
                std::shared_ptr<seal::SecretKey> m_secret_key;
                std::shared_ptr<seal::Encryptor> m_encryptor;
                std::shared_ptr<seal::Decryptor> m_decryptor;
                std::shared_ptr<seal::Evaluator> m_evaluator;
            };
        }
    }
}
