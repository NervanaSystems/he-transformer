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

#include "kernel/multiply.hpp"
#include "he_backend.hpp"
#include "he_seal_backend.hpp"
#include "kernel/ckks/multiply_ckks.hpp"
#include "kernel/ckks/negate_ckks.hpp"
#include "kernel/ckks/square_ckks.hpp"
#include "kernel/seal/multiply_seal.hpp"
#include "kernel/seal/negate_seal.hpp"
#include "kernel/seal/square_seal.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::multiply(const vector<shared_ptr<runtime::he::HECiphertext>>& arg0,
                                   const vector<shared_ptr<runtime::he::HECiphertext>>& arg1,
                                   vector<shared_ptr<runtime::he::HECiphertext>>& out,
                                   const element::Type& type,
                                   const shared_ptr<runtime::he::HEBackend>& he_backend,
                                   size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_multiply(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::scalar_multiply(const shared_ptr<runtime::he::HECiphertext>& arg0,
                                          const shared_ptr<runtime::he::HECiphertext>& arg1,
                                          shared_ptr<runtime::he::HECiphertext>& out,
                                          const element::Type& type,
                                          const shared_ptr<runtime::he::HEBackend>& he_backend)
{
    if (auto he_seal_backend =
            dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(he_backend))
    {
        shared_ptr<runtime::he::SealCiphertextWrapper> arg0_seal =
            dynamic_pointer_cast<runtime::he::SealCiphertextWrapper>(arg0);
        shared_ptr<runtime::he::SealCiphertextWrapper> arg1_seal =
            dynamic_pointer_cast<runtime::he::SealCiphertextWrapper>(arg1);
        shared_ptr<runtime::he::SealCiphertextWrapper> out_seal =
            dynamic_pointer_cast<runtime::he::SealCiphertextWrapper>(out);

        if (arg0_seal && arg1_seal && out_seal)
        {
            if (arg0_seal == arg1_seal)
            {
                kernel::seal::scalar_square(arg0_seal, out_seal, type, he_seal_backend);
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_seal);
            }
            else
            {
                kernel::seal::scalar_multiply(
                    arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
                he_seal_backend->get_evaluator()->relinearize_inplace(out_seal->m_ciphertext,
                                                              *(he_seal_backend->get_relin_key()));
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_seal);
            }
        }
        else
        {
            throw ngraph_error(
                "Multiply backend is SEAL, but arguments or outputs are not SealCiphertextWrapper");
        }
    }
    else if (auto he_ckks_backend =
                 dynamic_pointer_cast<runtime::he::he_ckks::HEHeaanBackend>(he_backend))
    {
        shared_ptr<runtime::he::HeaanCiphertextWrapper> arg0_ckks =
            dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(arg0);
        shared_ptr<runtime::he::HeaanCiphertextWrapper> arg1_ckks =
            dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(arg1);
        shared_ptr<runtime::he::HeaanCiphertextWrapper> out_ckks =
            dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(out);

        if (arg0_ckks && arg1_ckks && out_ckks)
        {
            if (arg0_ckks == arg1_ckks)
            {
                kernel::ckks::scalar_square(arg0_ckks, out_ckks, type, he_ckks_backend);
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_ckks);
            }
            else
            {
                kernel::ckks::scalar_multiply(
                    arg0_ckks, arg1_ckks, out_ckks, type, he_ckks_backend);
                he_ckks_backend->get_scheme()->reScaleByAndEqual(
                    out_ckks->m_ciphertext, he_ckks_backend->get_precision());
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_ckks);
            }
        }
        else
        {
            throw ngraph_error(
                "Multiply backend is HEAAN, but arguments or outputs are not "
                "HeaanCiphertextWrapper");
        }
    }
    else
    {
        throw ngraph_error("Multiply backend is neither SEAL nor HEAAN.");
    }
}

void runtime::he::kernel::multiply(const vector<shared_ptr<runtime::he::HECiphertext>>& arg0,
                                   const vector<shared_ptr<runtime::he::HEPlaintext>>& arg1,
                                   vector<shared_ptr<runtime::he::HECiphertext>>& out,
                                   const element::Type& type,
                                   const shared_ptr<runtime::he::HEBackend>& he_backend,
                                   size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_multiply(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::scalar_multiply(const shared_ptr<runtime::he::HECiphertext>& arg0,
                                          const shared_ptr<runtime::he::HEPlaintext>& arg1,
                                          shared_ptr<runtime::he::HECiphertext>& out,
                                          const element::Type& type,
                                          const shared_ptr<runtime::he::HEBackend>& he_backend)
{
    const string type_name = type.c_type_string();

    if (auto he_seal_backend =
            dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(he_backend))
    {
        shared_ptr<runtime::he::SealCiphertextWrapper> arg0_seal =
            dynamic_pointer_cast<runtime::he::SealCiphertextWrapper>(arg0);
        shared_ptr<runtime::he::SealPlaintextWrapper> arg1_seal =
            dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(arg1);
        shared_ptr<runtime::he::SealCiphertextWrapper> out_seal =
            dynamic_pointer_cast<runtime::he::SealCiphertextWrapper>(out);

        if (arg0_seal && arg1_seal && out_seal)
        {
            shared_ptr<runtime::he::HEPlaintext> one =
                he_seal_backend->get_valued_plaintext(1, type);
            shared_ptr<runtime::he::HEPlaintext> zero =
                he_seal_backend->get_valued_plaintext(0, type);
            shared_ptr<runtime::he::HEPlaintext> neg_one =
                he_seal_backend->get_valued_plaintext(-1, type);
            auto one_seal = dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(one);
            auto zero_seal = dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(zero);
            auto neg_one_seal = dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(neg_one);

            if (arg1_seal->m_plaintext == one_seal->m_plaintext)
            {
                out = arg0;
            }
            else if (arg1_seal->m_plaintext == neg_one_seal->m_plaintext)
            {
                kernel::seal::scalar_negate(arg0_seal, out_seal, type, he_seal_backend);
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_seal);
            }
            else if (arg1_seal->m_plaintext == zero_seal->m_plaintext)
            {
                out = he_seal_backend->create_valued_ciphertext(0, type);
            }
            else
            {
                kernel::seal::scalar_multiply(
                    arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
                // Don't relinearize, since plain multiplications never increase ciphertext size
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_seal);
            }
        }
        else
        {
            throw ngraph_error(
                "Multiply backend is SEAL, but arguments or outputs are not SealCiphertextWrapper");
        }
    }
    else if (auto he_ckks_backend =
                 dynamic_pointer_cast<runtime::he::he_ckks::HEHeaanBackend>(he_backend))
    {
        shared_ptr<runtime::he::HeaanCiphertextWrapper> arg0_ckks =
            dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(arg0);
        shared_ptr<runtime::he::HeaanPlaintextWrapper> arg1_ckks =
            dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(arg1);
        shared_ptr<runtime::he::HeaanCiphertextWrapper> out_ckks =
            dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(out);

        if (arg0_ckks && arg1_ckks && out_ckks)
        {
            /* auto neg_one = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(
                he_ckks_backend->get_valued_plaintext(-1, type));
            auto zero = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(
                he_ckks_backend->get_valued_plaintext(0, type));
            auto one = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(
                he_ckks_backend->get_valued_plaintext(1, type));

            if (arg1_ckks->m_plaintexts[0] == one->m_plaintexts[0])
            {
                out_ckks = arg0_ckks;
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_ckks);
            }
            else if (arg1_ckks->m_plaintexts[0] == neg_one->m_plaintexts[0])
            {
                kernel::ckks::scalar_negate(arg0_ckks, out_ckks, type, he_ckks_backend);
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_ckks);
            }
            else if (arg1_ckks->m_plaintexts[0] == zero->m_plaintexts[0])
            {
                // out = he_ckks_backend->get_valued_ciphertext(0, type);
                shared_ptr<runtime::he::HECiphertext> out_ckks = he_ckks_backend->create_valued_ciphertext(0, type, 1);

                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_ckks);

                // out = he_ckks_backend->create_valued_ciphertext(0, type, 1);

                const auto& tmp = neg_one;
                 std::shared_ptr<ngraph::runtime::he::HEPlaintext> test = tmp;
                he_ckks_backend->decrypt(test, out);
                NGRAPH_INFO << "Mult 0 => " << tmp->m_plaintexts[0];
            }
            else */
            {
                kernel::ckks::scalar_multiply(
                    arg0_ckks, arg1_ckks, out_ckks, type, he_ckks_backend);
                he_ckks_backend->get_scheme()->reScaleByAndEqual(
                    out_ckks->m_ciphertext, he_ckks_backend->get_precision());
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_ckks);
            }
        }
        else
        {
            throw ngraph_error(
                "Multiply backend is HEAAN, but arguments or outputs are not "
                "HeaanCiphertextWrapper");
        }
    }
    else
    {
        throw ngraph_error("Multiply backend is neither SEAL nor HEAAN.");
    }
}

void runtime::he::kernel::multiply(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg0,
                                   const vector<shared_ptr<runtime::he::HECiphertext>>& arg1,
                                   vector<shared_ptr<runtime::he::HECiphertext>>& out,
                                   const element::Type& type,
                                   const shared_ptr<runtime::he::HEBackend>& he_backend,
                                   size_t count)
{
    multiply(arg1, arg0, out, type, he_backend, count);
}

void runtime::he::kernel::scalar_multiply(const shared_ptr<runtime::he::HEPlaintext>& arg0,
                                          const shared_ptr<runtime::he::HECiphertext>& arg1,
                                          shared_ptr<runtime::he::HECiphertext>& out,
                                          const element::Type& type,
                                          const shared_ptr<runtime::he::HEBackend>& he_backend)
{
    scalar_multiply(arg1, arg0, out, type, he_backend);
}

void runtime::he::kernel::multiply(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg0,
                                   const vector<shared_ptr<runtime::he::HEPlaintext>>& arg1,
                                   vector<shared_ptr<runtime::he::HEPlaintext>>& out,
                                   const element::Type& type,
                                   const shared_ptr<runtime::he::HEBackend>& he_backend,
                                   size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_multiply(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::scalar_multiply(const shared_ptr<runtime::he::HEPlaintext>& arg0,
                                          const shared_ptr<runtime::he::HEPlaintext>& arg1,
                                          shared_ptr<runtime::he::HEPlaintext>& out,
                                          const element::Type& type,
                                          const shared_ptr<runtime::he::HEBackend>& he_backend)
{
    if (auto he_seal_backend =
            dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(he_backend))
    {
        shared_ptr<runtime::he::SealPlaintextWrapper> arg0_seal =
            dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(arg0);
        shared_ptr<runtime::he::SealPlaintextWrapper> arg1_seal =
            dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(arg1);
        shared_ptr<runtime::he::SealPlaintextWrapper> out_seal =
            dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(out);

        if (arg0_seal && arg1_seal && out_seal)
        {
            kernel::seal::scalar_multiply(arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
            out = dynamic_pointer_cast<runtime::he::HEPlaintext>(out_seal);
        }
        else
        {
            throw ngraph_error(
                "Multiply backend is SEAL, but arguments or outputs are not SealPlaintextWrapper");
        }
    }
    else if (auto he_ckks_backend =
                 dynamic_pointer_cast<runtime::he::he_ckks::HEHeaanBackend>(he_backend))
    {
        shared_ptr<runtime::he::HeaanPlaintextWrapper> arg0_ckks =
            dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(arg0);
        shared_ptr<runtime::he::HeaanPlaintextWrapper> arg1_ckks =
            dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(arg1);
        shared_ptr<runtime::he::HeaanPlaintextWrapper> out_ckks =
            dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(out);

        if (arg0_ckks && arg1_ckks && out_ckks)
        {
            kernel::ckks::scalar_multiply(
                arg0_ckks, arg1_ckks, out_ckks, type, he_ckks_backend);
            out = dynamic_pointer_cast<runtime::he::HEPlaintext>(out_ckks);
        }
        else
        {
            throw ngraph_error(
                "Multiply backend is HEAAN, but arguments or outputs are not "
                "HeaanPlaintextWrapper");
        }
    }
    else
    {
        throw ngraph_error("Multiply backend is neither SEAL nor HEAAN.");
    }
}
