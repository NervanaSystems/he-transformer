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
#include "he_ckks_backend.hpp"
#include "he_seal_backend.hpp"
#include "kernel/ckks/negate_ckks.hpp"
#include "kernel/ckks/subtract_ckks.hpp"
#include "kernel/seal/negate_seal.hpp"
#include "kernel/seal/subtract_seal.hpp"
#include "kernel/subtract.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::subtract(const vector<shared_ptr<runtime::he::HECiphertext>>& arg0,
                                   const vector<shared_ptr<runtime::he::HECiphertext>>& arg1,
                                   vector<shared_ptr<runtime::he::HECiphertext>>& out,
                                   const element::Type& type,
                                   const shared_ptr<runtime::he::HEBackend>& he_backend,
                                   size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_subtract(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::subtract(const vector<shared_ptr<runtime::he::HECiphertext>>& arg0,
                                   const vector<shared_ptr<runtime::he::HEPlaintext>>& arg1,
                                   vector<shared_ptr<runtime::he::HECiphertext>>& out,
                                   const element::Type& type,
                                   const shared_ptr<runtime::he::HEBackend>& he_backend,
                                   size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_subtract(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::subtract(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg0,
                                   const vector<shared_ptr<runtime::he::HECiphertext>>& arg1,
                                   vector<shared_ptr<runtime::he::HECiphertext>>& out,
                                   const element::Type& type,
                                   const shared_ptr<runtime::he::HEBackend>& he_backend,
                                   size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_subtract(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::subtract(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg0,
                                   const vector<shared_ptr<runtime::he::HEPlaintext>>& arg1,
                                   vector<shared_ptr<runtime::he::HEPlaintext>>& out,
                                   const element::Type& type,
                                   const shared_ptr<runtime::he::HEBackend>& he_backend,
                                   size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_subtract(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::scalar_subtract(const shared_ptr<runtime::he::HECiphertext>& arg0,
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
            kernel::seal::scalar_subtract(arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
            out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_seal);
        }
        else
        {
            throw ngraph_error(
                "Subtract backend is SEAL, but arguments or outputs are not SealCiphertextWrapper");
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
            kernel::ckks::scalar_subtract(
                arg0_ckks, arg1_ckks, out_ckks, type, he_ckks_backend);
            out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_ckks);
        }
        else
        {
            throw ngraph_error(
                "Subtract backend is HEAAN, but arguments or outputs are not "
                "HeaanCiphertextWrapper");
        }
    }
    else
    {
        throw ngraph_error("Subtract backend is neither SEAL nor HEAAN.");
    }
}

void runtime::he::kernel::scalar_subtract(const shared_ptr<runtime::he::HEPlaintext>& arg0,
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
            kernel::seal::scalar_subtract(arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
            out = dynamic_pointer_cast<runtime::he::HEPlaintext>(out_seal);
        }
        else
        {
            throw ngraph_error(
                "Subtract backend is SEAL, but arguments or outputs are not SealPlaintextWrapper");
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
            kernel::ckks::scalar_subtract(
                arg0_ckks, arg1_ckks, out_ckks, type, he_ckks_backend);
            out = dynamic_pointer_cast<runtime::he::HEPlaintext>(out_ckks);
        }
        else
        {
            throw ngraph_error(
                "Subtract backend is HEAAN, but arguments or outputs are not "
                "HeaanPlaintextWrapper");
        }
    }
    else
    {
        throw ngraph_error("Subtract backend is neither SEAL nor HEAAN.");
    }
}

void runtime::he::kernel::scalar_subtract(const shared_ptr<runtime::he::HECiphertext>& arg0,
                                          const shared_ptr<runtime::he::HEPlaintext>& arg1,
                                          shared_ptr<runtime::he::HECiphertext>& out,
                                          const element::Type& type,
                                          const shared_ptr<runtime::he::HEBackend>& he_backend)
{
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
            auto zero = dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(
                he_seal_backend->get_valued_plaintext(0, type));

            if (arg1_seal->m_plaintext == zero->m_plaintext)
            {
                out = arg0;
            }
            else
            {
                kernel::seal::scalar_subtract(
                    arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_seal);
            }
        }
        else
        {
            throw ngraph_error(
                "Subtract backend is SEAL, but arguments or outputs are not SealPlaintextWrapper");
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
            auto zero = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(
                he_ckks_backend->get_valued_plaintext(0, type));

            if (arg1_ckks->m_plaintexts == zero->m_plaintexts)
            {
                out = arg0;
            }
            else
            {
                kernel::ckks::scalar_subtract(
                    arg0_ckks, arg1_ckks, out_ckks, type, he_ckks_backend);
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_ckks);
            }
        }
        else
        {
            throw ngraph_error(
                "Subtract backend is HEAAN, but arguments or outputs are not "
                "HeaanPlaintextWrapper");
        }
    }
    else
    {
        throw ngraph_error("Subtract backend is neither SEAL nor HEAAN.");
    }
}

void runtime::he::kernel::scalar_subtract(const shared_ptr<runtime::he::HEPlaintext>& arg0,
                                          const shared_ptr<runtime::he::HECiphertext>& arg1,
                                          shared_ptr<runtime::he::HECiphertext>& out,
                                          const element::Type& type,
                                          const shared_ptr<runtime::he::HEBackend>& he_backend)
{
    if (auto he_seal_backend =
            dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(he_backend))
    {
        shared_ptr<runtime::he::SealPlaintextWrapper> arg0_seal =
            dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(arg0);
        shared_ptr<runtime::he::SealCiphertextWrapper> arg1_seal =
            dynamic_pointer_cast<runtime::he::SealCiphertextWrapper>(arg1);
        shared_ptr<runtime::he::SealCiphertextWrapper> out_seal =
            dynamic_pointer_cast<runtime::he::SealCiphertextWrapper>(out);

        if (arg0_seal && arg1_seal && out_seal)
        {
            auto zero = dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(
                he_seal_backend->get_valued_plaintext(0, type));

            if (arg0_seal->m_plaintext == zero->m_plaintext)
            {
                kernel::seal::scalar_negate(arg1_seal, out_seal, type, he_seal_backend);
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_seal);
            }
            else
            {
                kernel::seal::scalar_subtract(
                    arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_seal);
            }
        }
        else
        {
            throw ngraph_error(
                "Subtract backend is SEAL, but arguments or outputs are not SealPlaintextWrapper");
        }
    }
    else if (auto he_ckks_backend =
                 dynamic_pointer_cast<runtime::he::he_ckks::HEHeaanBackend>(he_backend))
    {
        shared_ptr<runtime::he::HeaanPlaintextWrapper> arg0_ckks =
            dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(arg0);
        shared_ptr<runtime::he::HeaanCiphertextWrapper> arg1_ckks =
            dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(arg1);
        shared_ptr<runtime::he::HeaanCiphertextWrapper> out_ckks =
            dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(out);

        if (arg0_ckks && arg1_ckks && out_ckks)
        {
            auto zero = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(
                he_ckks_backend->get_valued_plaintext(0, type));

            if (arg0_ckks->m_plaintexts == zero->m_plaintexts)
            {
                kernel::ckks::scalar_negate(arg1_ckks, out_ckks, type, he_ckks_backend);
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_ckks);
            }
            else
            {
                kernel::ckks::scalar_subtract(
                    arg0_ckks, arg1_ckks, out_ckks, type, he_ckks_backend);
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_ckks);
            }
        }
        else
        {
            throw ngraph_error(
                "Subtract backend is HEAAN, but arguments or outputs are not "
                "HeaanPlaintextWrapper");
        }
    }
    else
    {
        throw ngraph_error("Subtract backend is neither SEAL nor HEAAN.");
    }
}
