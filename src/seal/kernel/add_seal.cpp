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

#include "seal/kernel/add_seal.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void he_seal::kernel::scalar_add(const shared_ptr<he_seal::SealCiphertextWrapper>& arg0,
                                 const shared_ptr<he_seal::SealCiphertextWrapper>& arg1,
                                 shared_ptr<he_seal::SealCiphertextWrapper>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealBackend* he_seal_backend)
{
    if (auto he_seal_ckks_backend =
            dynamic_cast<const he_seal::HESealCKKSBackend*>(he_seal_backend))
    {
        double scale0 = arg0->m_ciphertext.scale();
        double scale1 = arg1->m_ciphertext.scale();

        size_t chain_ind0 = he_seal_ckks_backend->get_context()
                                ->context_data(arg0->m_ciphertext.parms_id())
                                ->chain_index();
        size_t chain_ind1 = he_seal_ckks_backend->get_context()
                                ->context_data(arg1->m_ciphertext.parms_id())
                                ->chain_index();

        if (scale0 < 0.999 * scale1 || scale0 > 1.001 * scale1)
        {
            NGRAPH_DEBUG << "Scale " << setw(10) << scale0 << " does not match scale " << scale1
                         << " in scalar add, ratio is " << scale0 / scale1;
        }
        if (scale0 != scale1)
        {
            arg0->m_ciphertext.scale() = arg1->m_ciphertext.scale();
        }

        while (chain_ind0 > chain_ind1)
        {
            // NGRAPH_INFO << "Mod switching " << chain_ind0 << " , " << chain_ind1;
            he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
                arg0->m_ciphertext, arg1->m_ciphertext.parms_id());
            chain_ind0 = he_seal_ckks_backend->get_context()
                             ->context_data(arg0->m_ciphertext.parms_id())
                             ->chain_index();
        }
        while (chain_ind1 > chain_ind0)
        {
            // NGRAPH_INFO << "Mod switching " << chain_ind0 << " , " << chain_ind1;
            he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
                arg1->m_ciphertext, arg0->m_ciphertext.parms_id());
            chain_ind1 = he_seal_ckks_backend->get_context()
                             ->context_data(arg1->m_ciphertext.parms_id())
                             ->chain_index();
        }
    }

    if (arg0 == out)
    {
        he_seal_backend->get_evaluator()->add_inplace(out->m_ciphertext, arg1->m_ciphertext);
    }
    else if (arg1 == out)
    {
        he_seal_backend->get_evaluator()->add_inplace(out->m_ciphertext, arg0->m_ciphertext);
    }
    else
    {
        he_seal_backend->get_evaluator()->add(
            arg0->m_ciphertext, arg1->m_ciphertext, out->m_ciphertext);
    }
}

void he_seal::kernel::scalar_add(const shared_ptr<he_seal::SealCiphertextWrapper>& arg0,
                                 const shared_ptr<he_seal::SealPlaintextWrapper>& arg1,
                                 shared_ptr<he_seal::SealCiphertextWrapper>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealBackend* he_seal_backend)
{
    if (arg0 == out)
    {
        he_seal_backend->get_evaluator()->add_plain_inplace(out->m_ciphertext, arg1->m_plaintext);
    }
    else
    {
        he_seal_backend->get_evaluator()->add_plain(
            arg0->m_ciphertext, arg1->m_plaintext, out->m_ciphertext);
    }
}

void he_seal::kernel::scalar_add(const shared_ptr<he_seal::SealPlaintextWrapper>& arg0,
                                 const shared_ptr<he_seal::SealPlaintextWrapper>& arg1,
                                 shared_ptr<he_seal::SealPlaintextWrapper>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealBackend* he_seal_backend)
{
    shared_ptr<HEPlaintext> out_he = dynamic_pointer_cast<HEPlaintext>(out);
    const string type_name = element_type.c_type_string();
    if (type_name == "float")
    {
        float x, y;
        he_seal_backend->decode(&x, arg0, element_type);
        he_seal_backend->decode(&y, arg1, element_type);
        float r = x + y;
        he_seal_backend->encode(out_he, &r, element_type);
    }
    else
    {
        throw ngraph_error("Unsupported element type " + type_name + " in add");
    }
    out = dynamic_pointer_cast<he_seal::SealPlaintextWrapper>(out_he);
}