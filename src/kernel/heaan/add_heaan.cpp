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

#include "kernel/heaan/add_heaan.hpp"
#include "he_heaan_backend.hpp"
#include "heaan_ciphertext_wrapper.hpp"
#include "heaan_plaintext_wrapper.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::heaan::scalar_add(const shared_ptr<runtime::he::HeaanCiphertextWrapper>& arg0,
                                            const shared_ptr<runtime::he::HeaanCiphertextWrapper>& arg1,
                                            shared_ptr<runtime::he::HeaanCiphertextWrapper>& out,
                                            const element::Type& type,
                                            shared_ptr<runtime::he::he_heaan::HEHeaanBackend> he_heaan_backend)
{
    if (out == arg0) // TOOD: Discover why this is needed? (dot.cpp needs this)
    {
        out->m_ciphertext =
            he_heaan_backend->get_scheme()->add(arg1->m_ciphertext, arg0->m_ciphertext);
    }
    else
    {
        out->m_ciphertext =
            he_heaan_backend->get_scheme()->add(arg0->m_ciphertext, arg1->m_ciphertext);
    }
}

void runtime::he::kernel::heaan::scalar_add(const shared_ptr<runtime::he::HeaanPlaintextWrapper>& arg0,
                                            const shared_ptr<runtime::he::HeaanPlaintextWrapper>& arg1,
                                            shared_ptr<runtime::he::HeaanPlaintextWrapper>& out,
                                            const element::Type& type,
                                            shared_ptr<runtime::he::he_heaan::HEHeaanBackend> he_heaan_backend)
{
    float x, y;
    he_heaan_backend->decode(&x, arg0, type);
    he_heaan_backend->decode(&y, arg1, type);
    float r = x + y;
    shared_ptr<runtime::he::HEPlaintext> out_he = dynamic_pointer_cast<runtime::he::HEPlaintext>(out);
    he_heaan_backend->encode(out_he, &r, type);
    out = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(out_he);
}

void runtime::he::kernel::heaan::scalar_add(const shared_ptr<runtime::he::HeaanCiphertextWrapper>& arg0,
                                            const shared_ptr<runtime::he::HeaanPlaintextWrapper>& arg1,
                                            shared_ptr<runtime::he::HeaanCiphertextWrapper>& out,
                                            const element::Type& type,
                                            shared_ptr<runtime::he::he_heaan::HEHeaanBackend> he_heaan_backend)
{
    out->m_ciphertext =
        he_heaan_backend->get_scheme()->addConst(arg0->m_ciphertext, arg1->m_plaintext);
}

void runtime::he::kernel::heaan::scalar_add(const shared_ptr<runtime::he::HeaanPlaintextWrapper>& arg0,
                                            const shared_ptr<runtime::he::HeaanCiphertextWrapper>& arg1,
                                            shared_ptr<runtime::he::HeaanCiphertextWrapper>& out,
                                            const element::Type& type,
                                            shared_ptr<runtime::he::he_heaan::HEHeaanBackend> he_heaan_backend)
{
    scalar_add(arg1, arg0, out, type, he_heaan_backend);
}
