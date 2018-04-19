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

#include <cstddef>

#include "he_cipher_tensor_view.hpp"
#include "seal/seal.h"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            void add(const HECipherTensorView* arg0, const HECipherTensorView* arg1, HECipherTensorView* out, size_t count)
            {
                char* arg0_dp = (char*)(arg0->get_data_ptr());
                char* arg1_dp = (char*)(arg1->get_data_ptr());
                char* out0_dp = (char*)(arg0->get_data_ptr());

                size_t offset = 0;
                for(size_t i = 0; i < count; ++i)
                {
                    seal::Ciphertext* arg0 = reinterpret_cast<seal::Ciphertext*>(arg0_dp[offset]);
                    seal::Ciphertext* arg1 = reinterpret_cast<seal::Ciphertext*>(arg1_dp[offset]);
                    seal::Ciphertext* out0 = reinterpret_cast<seal::Ciphertext*>(out0_dp[offset]);

                    arg0->m_he_backend->evaluator->add(*arg0, *arg1, *out0);


                }
                std::cout << "count " << count << std::endl;

                offset += sizeof(seal::Ciphertext);
            }
        }
    }
}
