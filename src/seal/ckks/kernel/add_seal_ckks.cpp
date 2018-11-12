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


#include <iomanip>
#include "seal/ckks/kernel/add_seal_ckks.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void he_seal::ckks::kernel::scalar_add_ckks(const shared_ptr<const he_seal::SealCiphertextWrapper>& arg0,
                                 const shared_ptr<const he_seal::SealCiphertextWrapper>& arg1,
                                 shared_ptr<he_seal::SealCiphertextWrapper>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealCKKSBackend* he_seal_ckks_backend)
{
    auto arg0_scaled = make_shared<he_seal::SealCiphertextWrapper>(arg0->m_ciphertext);
    auto arg1_scaled = make_shared<he_seal::SealCiphertextWrapper>(arg1->m_ciphertext);

    double scale0 = arg0->m_ciphertext.scale();
    double scale1 = arg1->m_ciphertext.scale();

    size_t chain_ind0 = he_seal_ckks_backend->get_context()
                            ->context_data(arg0->m_ciphertext.parms_id())
                            ->chain_index();
    size_t chain_ind1 = he_seal_ckks_backend->get_context()
                            ->context_data(arg1->m_ciphertext.parms_id())
                            ->chain_index();

    if (scale0 < 0.99 * scale1 || scale0 > 1.01 * scale1)
    {
        // NGRAPH_DEBUG isn't thread-safe until ngraph commit #1977
        // https://github.com/NervanaSystems/ngraph/commit/ee6444ed39864776c8ce9a406eee9275382a88bb
        // so we comment it out.
        // TODO: use NGRAPH_DEBUG at next ngraph version
        NGRAPH_WARN << "Scale " << setw(10) << scale0 << " does not match scale " << scale1
                        << " in scalar add, ratio is " << scale0 / scale1;
    }
    if (scale0 != scale1)
    {
        arg0_scaled->m_ciphertext.scale() = arg1_scaled->m_ciphertext.scale();
    }

    if (chain_ind0 > chain_ind1)
    {
        // NGRAPH_INFO << "Chain switching " << chain_ind0 << ", " << chain_ind1;
        he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
            arg0_scaled->m_ciphertext, arg1_scaled->m_ciphertext.parms_id());
        chain_ind0 = he_seal_ckks_backend->get_context()
                            ->context_data(arg0_scaled->m_ciphertext.parms_id())
                            ->chain_index();
    }
    else if (chain_ind1 > chain_ind0)
    {
        // NGRAPH_INFO << "Chain switching " << chain_ind0 << ", " << chain_ind1;
        he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
            arg1_scaled->m_ciphertext, arg0_scaled->m_ciphertext.parms_id());
        chain_ind1 = he_seal_ckks_backend->get_context()
                            ->context_data(arg1_scaled->m_ciphertext.parms_id())
                            ->chain_index();
    }
    NGRAPH_ASSERT(chain_ind1 == chain_ind0) << "Chain moduli are different";

    if (arg0 == out)
    {
        he_seal_ckks_backend->get_evaluator()->add_inplace(arg0_scaled->m_ciphertext, arg1_scaled->m_ciphertext);
        out = arg0_scaled;
    }
    else if (arg1 == out)
    {
        he_seal_ckks_backend->get_evaluator()->add_inplace(arg1_scaled->m_ciphertext, arg0_scaled->m_ciphertext);
        out = arg1_scaled;
    }
    else
    {
        he_seal_ckks_backend->get_evaluator()->add(
            arg0_scaled->m_ciphertext, arg1_scaled->m_ciphertext, out->m_ciphertext);
    }
}

void he_seal::ckks::kernel::scalar_add_ckks(const shared_ptr<const he_seal::SealCiphertextWrapper>& arg0,
                                 const shared_ptr<const he_seal::SealPlaintextWrapper>& arg1,
                                 shared_ptr<he_seal::SealCiphertextWrapper>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealCKKSBackend* he_seal_ckks_backend)
{
    // TODO: enable with different scale / modulus chain
    if (arg0 == out)
    {
        he_seal_ckks_backend->get_evaluator()->add_plain_inplace(out->m_ciphertext, arg1->m_plaintext);
    }
    else
    {
        he_seal_ckks_backend->get_evaluator()->add_plain(
            arg0->m_ciphertext, arg1->m_plaintext, out->m_ciphertext);
    }
}

void he_seal::ckks::kernel::scalar_add_ckks(const shared_ptr<const he_seal::SealPlaintextWrapper>& arg0,
                                 const shared_ptr<const he_seal::SealCiphertextWrapper>& arg1,
                                 shared_ptr<he_seal::SealCiphertextWrapper>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealCKKSBackend* he_seal_ckks_backend)
{
    he_seal::ckks::kernel::scalar_add_ckks(arg1, arg0, out, element_type, he_seal_ckks_backend);
}
