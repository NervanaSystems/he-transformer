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

#include "seal/seal.h"

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
                void add(const vector<shared_ptr<seal::Ciphertext>>& arg0,
                         const vector<shared_ptr<seal::Ciphertext>>& arg1,
                         vector<shared_ptr<seal::Ciphertext>>& out,
                         shared_ptr<HEBackend> he_backend,
                         size_t count);

                void add(const shared_ptr<seal::Ciphertext>& arg0,
                         const shared_ptr<seal::Ciphertext>& arg1,
                         shared_ptr<seal::Ciphertext>& out,
                         shared_ptr<HEBackend> he_backend);

                void add(const vector<shared_ptr<seal::Ciphertext>>& arg0,
                         const vector<shared_ptr<seal::Plaintext>>& arg1,
                         vector<shared_ptr<seal::Ciphertext>>& out,
                         shared_ptr<HEBackend> he_backend,
                         size_t count);

                void add(const vector<shared_ptr<seal::Plaintext>>& arg0,
                         const vector<shared_ptr<seal::Ciphertext>>& arg1,
                         vector<shared_ptr<seal::Ciphertext>>& out,
                         shared_ptr<HEBackend> he_backend,
                         size_t count);

                void add(const vector<shared_ptr<seal::Plaintext>>& arg0,
                         const vector<shared_ptr<seal::Plaintext>>& arg1,
                         vector<shared_ptr<seal::Plaintext>>& out,
                         const element::Type& type,
                         shared_ptr<HEBackend> he_backend,
                         size_t count);

                void add(const shared_ptr<seal::Plaintext>& arg0,
                         const shared_ptr<seal::Plaintext>& arg1,
                         shared_ptr<seal::Plaintext>& out,
                         const element::Type& type,
                         shared_ptr<HEBackend> he_backend);
            }
        }
    }
}
