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
#include <vector>

#include "heaan_ciphertext_wrapper.hpp"

namespace ngraph
{
    namespace element
    {
        class Type;
    }
    namespace runtime
    {
        namespace he
        {
            class HEBackend;

            namespace kernel
            {
                void subtract(const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& arg0,
                              const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& arg1,
                              std::vector<std::shared_ptr<runtime::he::HECiphertext>>& out,
                              const element::Type& type,
                              const std::shared_ptr<runtime::he::HEBackend> he_backend,
                              size_t count);

                void subtract(const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& arg0,
                              const std::vector<std::shared_ptr<runtime::he::HEPlaintext>>& arg1,
                              std::vector<std::shared_ptr<runtime::he::HECiphertext>>& out,
                              const element::Type& type,
                              const std::shared_ptr<runtime::he::HEBackend> he_backend,
                              size_t count);

                void subtract(const std::vector<std::shared_ptr<runtime::he::HEPlaintext>>& arg0,
                              const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& arg1,
                              std::vector<std::shared_ptr<runtime::he::HECiphertext>>& out,
                              const element::Type& type,
                              const std::shared_ptr<runtime::he::HEBackend> he_backend,
                              size_t count);

                void subtract(const std::vector<std::shared_ptr<runtime::he::HEPlaintext>>& arg0,
                              const std::vector<std::shared_ptr<runtime::he::HEPlaintext>>& arg1,
                              std::vector<std::shared_ptr<runtime::he::HEPlaintext>>& out,
                              const element::Type& type,
                              const std::shared_ptr<runtime::he::HEBackend> he_backend,
                              size_t count);

                void scalar_subtract(const std::shared_ptr<runtime::he::HECiphertext>& arg0,
                                     const std::shared_ptr<runtime::he::HECiphertext>& arg1,
                                     std::shared_ptr<runtime::he::HECiphertext>& out,
                                     const element::Type& type,
                                     const std::shared_ptr<runtime::he::HEBackend> he_backend);

                void scalar_subtract(const std::shared_ptr<runtime::he::HECiphertext>& arg0,
                                     const std::shared_ptr<runtime::he::HEPlaintext>& arg1,
                                     std::shared_ptr<runtime::he::HECiphertext>& out,
                                     const element::Type& type,
                                     const std::shared_ptr<runtime::he::HEBackend> he_backend);

                void scalar_subtract(const std::shared_ptr<runtime::he::HEPlaintext>& arg0,
                                     const std::shared_ptr<runtime::he::HECiphertext>& arg1,
                                     std::shared_ptr<runtime::he::HECiphertext>& out,
                                     const element::Type& type,
                                     const std::shared_ptr<runtime::he::HEBackend> he_backend);

                void scalar_subtract(const std::shared_ptr<runtime::he::HEPlaintext>& arg0,
                                     const std::shared_ptr<runtime::he::HEPlaintext>& arg1,
                                     std::shared_ptr<runtime::he::HEPlaintext>& out,
                                     const element::Type& type,
                                     const std::shared_ptr<runtime::he::HEBackend> he_backend);
            }
        }
    }
}
