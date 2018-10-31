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

#include "seal/kernel/multiply_seal.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"

using namespace std;
using namespace ngraph::runtime::he;

void he_seal::kernel::scalar_multiply(const shared_ptr<he_seal::SealCiphertextWrapper>& arg0,
                                      const shared_ptr<he_seal::SealCiphertextWrapper>& arg1,
                                      shared_ptr<he_seal::SealCiphertextWrapper>& out,
                                      const element::Type& type,
                                      const runtime::he::he_seal::HESealBackend* he_seal_backend)
{
    if ((arg0 == arg1) && (arg1 == out))
    {
        he_seal_backend->get_evaluator()->square_inplace(out->m_ciphertext);
    }
    else if (arg1 == arg0)
    {
        he_seal_backend->get_evaluator()->square(arg1->m_ciphertext, out->m_ciphertext);
    }
    else if (arg0 == out)
    {
        he_seal_backend->get_evaluator()->multiply_inplace(out->m_ciphertext, arg1->m_ciphertext);
    }
    else if (arg1 == out)
    {
        he_seal_backend->get_evaluator()->multiply_inplace(out->m_ciphertext, arg0->m_ciphertext);
    }
    else
    {
        he_seal_backend->get_evaluator()->multiply(
            arg0->m_ciphertext, arg1->m_ciphertext, out->m_ciphertext);
    }

    he_seal_backend->get_evaluator()->relinearize_inplace(out->m_ciphertext,
                                                          *(he_seal_backend->get_relin_keys()));

    if (auto he_seal_ckks_backend =
            dynamic_cast<const he_seal::HESealCKKSBackend*>(he_seal_backend))
    {
        he_seal_ckks_backend->get_evaluator()->rescale_to_next_inplace(out->m_ciphertext);
    }
}

void he_seal::kernel::scalar_multiply(const shared_ptr<he_seal::SealCiphertextWrapper>& arg0,
                                      const shared_ptr<he_seal::SealPlaintextWrapper>& arg1,
                                      shared_ptr<he_seal::SealCiphertextWrapper>& out,
                                      const element::Type& type,
                                      const runtime::he::he_seal::HESealBackend* he_seal_backend)
{
    if (auto he_seal_ckks_backend =
            dynamic_cast<const he_seal::HESealCKKSBackend*>(he_seal_backend))
    {
        size_t chain_ind0 = he_seal_ckks_backend->get_context()
                                ->context_data(arg0->m_ciphertext.parms_id())
                                ->chain_index();
        size_t chain_ind1 = he_seal_ckks_backend->get_context()
                                ->context_data(arg1->m_plaintext.parms_id())
                                ->chain_index();

        while (chain_ind0 > chain_ind1) // TODO: switch to if-statement
        {
            // NGRAPH_INFO << "Mod switching " << chain_ind0 << " , " << chain_ind1;
            he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
                arg0->m_ciphertext, arg1->m_plaintext.parms_id());
            chain_ind0 = he_seal_ckks_backend->get_context()
                             ->context_data(arg0->m_ciphertext.parms_id())
                             ->chain_index();
        }
        while (chain_ind1 > chain_ind0) // TODO: switch to if-statement
        {
            // NGRAPH_INFO << "Mod switching " << chain_ind0 << " , " << chain_ind1;
            he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
                arg1->m_plaintext, arg0->m_ciphertext.parms_id());
            chain_ind1 = he_seal_ckks_backend->get_context()
                             ->context_data(arg1->m_plaintext.parms_id())
                             ->chain_index();
        }
    }

    if (arg0 == out)
    {
        he_seal_backend->get_evaluator()->multiply_plain_inplace(out->m_ciphertext,
                                                                 arg1->m_plaintext);
    }
    else
    {
        he_seal_backend->get_evaluator()->multiply_plain(
            arg0->m_ciphertext, arg1->m_plaintext, out->m_ciphertext);
    }

    he_seal_backend->get_evaluator()->relinearize_inplace(out->m_ciphertext,
                                                          *(he_seal_backend->get_relin_keys()));

    if (auto he_seal_ckks_backend =
            dynamic_cast<const he_seal::HESealCKKSBackend*>(he_seal_backend))
    {
        // TODO: rescale only if needed? Check mod switching?
        he_seal_ckks_backend->get_evaluator()->rescale_to_next_inplace(out->m_ciphertext);
    }
}

void he_seal::kernel::scalar_multiply(const shared_ptr<he_seal::SealPlaintextWrapper>& arg0,
                                      const shared_ptr<he_seal::SealPlaintextWrapper>& arg1,
                                      shared_ptr<he_seal::SealPlaintextWrapper>& out,
                                      const element::Type& type,
                                      const runtime::he::he_seal::HESealBackend* he_seal_backend)
{
    shared_ptr<runtime::he::HEPlaintext> out_he =
        dynamic_pointer_cast<runtime::he::HEPlaintext>(out);
    const string type_name = type.c_type_string();
    if (type_name == "float")
    {
        float x, y;
        he_seal_backend->decode(&x, arg0, type);
        he_seal_backend->decode(&y, arg1, type);
        float r = x * y;
        he_seal_backend->encode(out_he, &r, type);
    }
    else
    {
        throw ngraph_error("Unsupported type " + type_name + " in multiply");
    }
    out = dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(out_he);
}