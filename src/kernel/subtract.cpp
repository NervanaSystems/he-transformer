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

#include <vector>

#include "he_backend.hpp"
#include "he_seal_backend.hpp"
#include "kernel/subtract.hpp"
#include "seal/seal.h"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::subtract(const vector<shared_ptr<he::HECiphertext>>& arg0,
                                   const vector<shared_ptr<he::HECiphertext>>& arg1,
                                   vector<shared_ptr<he::HECiphertext>>& out,
                                   shared_ptr<HEBackend> he_backend,
                                   size_t count)
{
    auto he_seal_backend = dynamic_pointer_cast<HESealBackend>(he_backend);
    if (!he_seal_backend)
    {
        throw ngraph_error("HE backend not seal type");
    }
    for (size_t i = 0; i < count; ++i)
    {
        auto arg0i = dynamic_pointer_cast<SealCiphertextWrapper>(arg0[i]);
        auto arg1i = dynamic_pointer_cast<SealCiphertextWrapper>(arg1[i]);
        auto outi = dynamic_pointer_cast<SealCiphertextWrapper>(out[i]);
        if (arg0i != nullptr && arg1i != nullptr && outi != nullptr)
        {
            he_seal_backend->get_evaluator()->sub(
                arg0i->m_ciphertext, arg1i->m_ciphertext, outi->m_ciphertext);
        }
        else
        {
            throw ngraph_error("HE seal backend passed non-seal ciphertexts");
        }
    }
}

void runtime::he::kernel::subtract(const vector<shared_ptr<he::HECiphertext>>& arg0,
                                   const vector<shared_ptr<he::HEPlaintext>>& arg1,
                                   vector<shared_ptr<he::HECiphertext>>& out,
                                   shared_ptr<HEBackend> he_backend,
                                   size_t count)
{
    auto he_seal_backend = dynamic_pointer_cast<HESealBackend>(he_backend);
    if (!he_seal_backend)
    {
        throw ngraph_error("HE backend not seal type");
    }
    for (size_t i = 0; i < count; ++i)
    {
        auto arg0i = dynamic_pointer_cast<SealCiphertextWrapper>(arg0[i]);
        auto arg1i = dynamic_pointer_cast<SealPlaintextWrapper>(arg1[i]);
        auto outi = dynamic_pointer_cast<SealCiphertextWrapper>(out[i]);
        he_seal_backend->get_evaluator()->sub_plain(
            arg0i->m_ciphertext, arg1i->m_plaintext, outi->m_ciphertext);
    }
}
