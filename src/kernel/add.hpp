//*****************************************************************************
// Copyright 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>
#include <vector>

#include "he_backend.hpp"
#include "he_ciphertext.hpp"
#include "he_plaintext.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            namespace kernel
            {
                void scalar_add(const std::shared_ptr<runtime::he::HECiphertext>& arg0,
                                const std::shared_ptr<runtime::he::HECiphertext>& arg1,
                                std::shared_ptr<runtime::he::HECiphertext>& out,
                                const element::Type& element_type,
                                const runtime::he::HEBackend* he_backend);

                void scalar_add(const std::shared_ptr<runtime::he::HECiphertext>& arg0,
                                const std::shared_ptr<runtime::he::HEPlaintext>& arg1,
                                std::shared_ptr<runtime::he::HECiphertext>& out,
                                const element::Type& element_type,
                                const runtime::he::HEBackend* he_backend);

                void scalar_add(const std::shared_ptr<runtime::he::HEPlaintext>& arg0,
                                const std::shared_ptr<runtime::he::HECiphertext>& arg1,
                                std::shared_ptr<runtime::he::HECiphertext>& out,
                                const element::Type& element_type,
                                const runtime::he::HEBackend* he_backend);

                void scalar_add(const std::shared_ptr<runtime::he::HEPlaintext>& arg0,
                                const std::shared_ptr<runtime::he::HEPlaintext>& arg1,
                                std::shared_ptr<runtime::he::HEPlaintext>& out,
                                const element::Type& element_type,
                                const runtime::he::HEBackend* he_backend);

                template <typename S, typename T, typename V>
                void add(const std::vector<std::shared_ptr<S>>& arg0,
                         const std::vector<std::shared_ptr<T>>& arg1,
                         std::vector<std::shared_ptr<V>>& out,
                         const element::Type& element_type,
                         const runtime::he::HEBackend* he_backend,
                         size_t count)
                {
#pragma omp parallel for
                    for (size_t i = 0; i < count; ++i)
                    {
                        scalar_add(arg0[i], arg1[i], out[i], element_type, he_backend);
                    }
                }
            }
        }
    }
}
