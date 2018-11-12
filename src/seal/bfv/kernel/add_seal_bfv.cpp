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

#include "seal/bfv/kernel/add_seal_bfv.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void he_seal::bfv::kernel::scalar_add_bfv(const shared_ptr<const he_seal::SealCiphertextWrapper>& arg0,
                                 const shared_ptr<const he_seal::SealCiphertextWrapper>& arg1,
                                 shared_ptr<he_seal::SealCiphertextWrapper>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealBFVBackend* he_seal_bfv_backend)
{
    if (arg0 == out)
    {
        he_seal_bfv_backend->get_evaluator()->add_inplace(out->m_ciphertext, arg1->m_ciphertext);
    }
    else if (arg1 == out)
    {
        he_seal_bfv_backend->get_evaluator()->add_inplace(out->m_ciphertext, arg0->m_ciphertext);
    }
    else
    {
        he_seal_bfv_backend->get_evaluator()->add(
            arg0->m_ciphertext, arg1->m_ciphertext, out->m_ciphertext);
    }
}

void he_seal::bfv::kernel::scalar_add_bfv(const shared_ptr<const he_seal::SealCiphertextWrapper>& arg0,
                                 const shared_ptr<const he_seal::SealPlaintextWrapper>& arg1,
                                 shared_ptr<he_seal::SealCiphertextWrapper>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealBFVBackend* he_seal_bfv_backend)
{
    if (arg0 == out)
    {
        he_seal_bfv_backend->get_evaluator()->add_plain_inplace(out->m_ciphertext, arg1->m_plaintext);
    }
    else
    {
        he_seal_bfv_backend->get_evaluator()->add_plain(
            arg0->m_ciphertext, arg1->m_plaintext, out->m_ciphertext);
    }
}

void he_seal::bfv::kernel::scalar_add_bfv(const shared_ptr<const he_seal::SealPlaintextWrapper>& arg0,
                                 const shared_ptr<const he_seal::SealCiphertextWrapper>& arg1,
                                 shared_ptr<he_seal::SealCiphertextWrapper>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealBFVBackend* he_seal_bfv_backend)
{
    he_seal::bfv::kernel::scalar_add_bfv(arg1, arg0, out, element_type, he_seal_bfv_backend);
}

void he_seal::bfv::kernel::scalar_add_bfv(const shared_ptr<he_seal::SealPlaintextWrapper>& arg0,
                                 const shared_ptr<he_seal::SealPlaintextWrapper>& arg1,
                                 shared_ptr<he_seal::SealPlaintextWrapper>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealBFVBackend* he_seal_bfv_backend)
{
    shared_ptr<HEPlaintext> out_he = dynamic_pointer_cast<HEPlaintext>(out);
    const string type_name = element_type.c_type_string();
    if (type_name == "float")
    {
        float x, y;
        he_seal_bfv_backend->decode(&x, arg0, element_type);
        he_seal_bfv_backend->decode(&y, arg1, element_type);
        float r = x + y;
        he_seal_bfv_backend->encode(out_he, &r, element_type);
    }
    else
    {
        throw ngraph_error("Unsupported element type " + type_name + " in add");
    }
    out = dynamic_pointer_cast<he_seal::SealPlaintextWrapper>(out_he);
}