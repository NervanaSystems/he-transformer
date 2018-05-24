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

#include <vector>

#include "heaan_ciphertext_wrapper.hpp"
#include "he_heaan_backend.hpp"

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
            namespace he_heaan
            {
                class HEHeaanBackend;
            }

            namespace kernel
            {
                namespace heaan
                {
                    void add(const std::vector<std::shared_ptr<he::HeaanCiphertextWrapper>>& arg0,
                            const std::vector<std::shared_ptr<he::HeaanCiphertextWrapper>>& arg1,
                            std::vector<std::shared_ptr<he::HeaanCiphertextWrapper>>& out,
                            std::shared_ptr<he_heaan::HEHeaanBackend> he_heaan_backend,
                            size_t count);

                    void add(const std::vector<std::shared_ptr<he::HeaanCiphertextWrapper>>& arg0,
                            const std::vector<std::shared_ptr<he::HeaanPlaintextWrapper>>& arg1,
                            std::vector<std::shared_ptr<he::HeaanCiphertextWrapper>>& out,
                            std::shared_ptr<he_heaan::HEHeaanBackend> he_heaan_backend,
                            size_t count);

                    void add(const std::vector<std::shared_ptr<he::HeaanPlaintextWrapper>>& arg0,
                            const std::vector<std::shared_ptr<he::HeaanCiphertextWrapper>>& arg1,
                            std::vector<std::shared_ptr<he::HeaanCiphertextWrapper>>& out,
                            std::shared_ptr<he_heaan::HEHeaanBackend> he_heaan_backend,
                            size_t count);

                    void add(const std::vector<std::shared_ptr<he::HeaanPlaintextWrapper>>& arg0,
                            const std::vector<std::shared_ptr<he::HeaanPlaintextWrapper>>& arg1,
                            std::vector<std::shared_ptr<he::HeaanPlaintextWrapper>>& out,
                            const element::Type& type,
                            std::shared_ptr<he_heaan::HEHeaanBackend> he_heaan_backend,
                            size_t count);

                    void scalar_add(const std::shared_ptr<he::HeaanCiphertextWrapper>& arg0,
                            const std::shared_ptr<he::HeaanCiphertextWrapper>& arg1,
                            std::shared_ptr<he::HeaanCiphertextWrapper>& out,
                            std::shared_ptr<he_heaan::HEHeaanBackend> he_heaan_backend);

                    void scalar_add(const std::shared_ptr<he::HeaanPlaintextWrapper>& arg0,
                            const std::shared_ptr<he::HeaanPlaintextWrapper>& arg1,
                            std::shared_ptr<he::HeaanPlaintextWrapper>& out,
                            const element::Type& type,
                            std::shared_ptr<he_heaan::HEHeaanBackend> he_heaan_backend);
                }
            }
        }
    }
}
