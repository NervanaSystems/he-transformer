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

#include "he_ckks_backend.hpp"
#include "he_bfv_backend.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/seal.h"

using namespace std;
using namespace ngraph;

void runtime::he::he_seal::kernel::scalar_add(const std::shared_ptr<runtime::he::he_seal::SealCiphertextWrapper>& arg0,
                            const std::shared_ptr<runtime::he::he_seal::SealCiphertextWrapper>& arg1,
                            std::shared_ptr<runtime::he::he_seal::SealCiphertextWrapper>& out,
                            const element::Type& type,
                            const std::shared_ptr<runtime::he::he_seal::HESealBackend>& he_seal_backend);

{
    if ((arg0 == arg1) && ((arg1 == out))
    {
       he_seal_backend->get_evaluator->square_inplace(out->m_ciphertext);
    }
    else if (arg1 == arg0)
    {
       he_seal_backend->get_evaluator->square(arg1->m_ciphertext, out->m_ciphertext);
    }
    else if (arg0 == out)
    {
       he_seal_backend->get_evaluator->add_inplace(out->m_ciphertext, arg1->m_ciphertext);
    }
    else if (arg1 == out)
    {
        he_seal_backend->get_evaluator->add_inplace(arg1->m_ciphertext, out->m_ciphertext);
    }
    else
    {
        he_seal_backend->get_evaluator->add(arg0->m_ciphertext, arg1->m_ciphertext, out->m_ciphertext);
    }
}

void runtime::he::he_seal::kernel::scalar_add(const std::shared_ptr<runtime::he::he_seal::SealCiphertextWrapper>& arg0,
                            const std::shared_ptr<runtime::he::he_seal::SealPlaintextWrapper>& arg1,
                            std::shared_ptr<runtime::he::he_seal::SealCiphertextWrapper>& out,
                            const element::Type& type,
                            const std::shared_ptr<runtime::he::he_seal::HESealBackend>& he_seal_backend);

{
    if (arg1 == out)
    {
        he_seal_backend->get_evaluator->add_plain_inplace(out->m_ciphertext, arg1->m_ciphertext);
    }
    else
    {
        he_seal_backend->get_evaluator->add_plain(arg0->m_ciphertext, arg1->m_ciphertext, out->m_ciphertext);
    }
}

void runtime::he::he_seal::kernel::scalar_add(const std::shared_ptr<runtime::he::he_seal::SealPlaintextWrapper>& arg0,
                            const std::shared_ptr<runtime::he::he_seal::SealPlaintextWrapper>& arg1,
                            std::shared_ptr<runtime::he::he_seal::SealPlaintextWrapper>& out,
                            const element::Type& type,
                            const std::shared_ptr<runtime::he::he_seal::HESealBackend>& he_seal_backend);

{
    throw ngraph_error("Scalar add not implemented");
}