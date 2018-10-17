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

#include "he_heaan_backend.hpp"
#include "he_seal_backend.hpp"
#include "kernel/add.hpp"
#include "kernel/heaan/add_heaan.hpp"
#include "kernel/seal/add_seal.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::add(const vector<shared_ptr<runtime::he::HECiphertext>>& arg0,
                              const vector<shared_ptr<runtime::he::HECiphertext>>& arg1,
                              vector<shared_ptr<runtime::he::HECiphertext>>& out,
                              const element::Type& type,
                              const shared_ptr<runtime::he::HEBackend>& he_backend,
                              size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_add(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::add(const vector<shared_ptr<runtime::he::HECiphertext>>& arg0,
                              const vector<shared_ptr<runtime::he::HEPlaintext>>& arg1,
                              vector<shared_ptr<runtime::he::HECiphertext>>& out,
                              const element::Type& type,
                              const shared_ptr<runtime::he::HEBackend>& he_backend,
                              size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_add(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::add(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg0,
                              const vector<shared_ptr<runtime::he::HECiphertext>>& arg1,
                              vector<shared_ptr<runtime::he::HECiphertext>>& out,
                              const element::Type& type,
                              const shared_ptr<runtime::he::HEBackend>& he_backend,
                              size_t count)
{
    add(arg1, arg0, out, type, he_backend, count);
}

void runtime::he::kernel::add(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg0,
                              const vector<shared_ptr<runtime::he::HEPlaintext>>& arg1,
                              vector<shared_ptr<runtime::he::HEPlaintext>>& out,
                              const element::Type& type,
                              const shared_ptr<runtime::he::HEBackend>& he_backend,
                              size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_add(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::scalar_add(const shared_ptr<runtime::he::HECiphertext>& arg0,
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
            kernel::seal::scalar_add(arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
            out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_seal);
        }
        else
        {
            throw ngraph_error(
                "Add backend is SEAL, but arguments or outputs are not SealCiphertextWrapper");
        }
    }
    else if (auto he_heaan_backend =
                 dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(he_backend))
    {
        shared_ptr<runtime::he::HeaanCiphertextWrapper> arg0_heaan =
            dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(arg0);
        shared_ptr<runtime::he::HeaanCiphertextWrapper> arg1_heaan =
            dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(arg1);
        shared_ptr<runtime::he::HeaanCiphertextWrapper> out_heaan =
            dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(out);

        if (arg0_heaan && arg1_heaan && out_heaan)
        {
            std::shared_ptr<runtime::he::HEPlaintext> plain0_val = make_shared<runtime::he::HeaanPlaintextWrapper>();
            he_heaan_backend->decrypt(plain0_val, arg0_heaan);
            float arg0_plain = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(plain0_val)->m_plaintexts[0];

            std::shared_ptr<runtime::he::HEPlaintext> plain1_val = make_shared<runtime::he::HeaanPlaintextWrapper>();
            he_heaan_backend->decrypt(plain1_val, arg1_heaan);
            float arg1_plain = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(plain1_val)->m_plaintexts[0];

            kernel::heaan::scalar_add(arg0_heaan, arg1_heaan, out_heaan, type, he_heaan_backend);
            out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_heaan);

            std::shared_ptr<runtime::he::HEPlaintext> plain_out = make_shared<runtime::he::HeaanPlaintextWrapper>();
            he_heaan_backend->decrypt(plain_out, out_heaan);
            float plain_out_val = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(plain_out)->m_plaintexts[0];

            NGRAPH_INFO << "Adding (cipher) " << arg0_plain << " + (cipher) " << arg1_plain
                << " => " << plain_out_val;

            if (plain_out_val > 1e50 || plain_out_val < -1e50)
            {
                NGRAPH_INFO << "plain val " << plain_out_val << " incorrect!";
                exit(0);
            }
        }
        else
        {
            throw ngraph_error(
                "Add backend is HEAAN, but arguments or outputs are not HeaanCiphertextWrapper");
        }
    }
    else
    {
        throw ngraph_error("Add backend is neither SEAL nor HEAAN.");
    }
}

void runtime::he::kernel::scalar_add(const shared_ptr<runtime::he::HEPlaintext>& arg0,
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
            kernel::seal::scalar_add(arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
            out = dynamic_pointer_cast<runtime::he::HEPlaintext>(out_seal);
        }
        else
        {
            throw ngraph_error(
                "Add backend is SEAL, but arguments or outputs are not SealPlaintextWrapper.:");
        }
    }
    else if (auto he_heaan_backend =
                 dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(he_backend))
    {
        shared_ptr<runtime::he::HeaanPlaintextWrapper> arg0_heaan =
            dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(arg0);
        shared_ptr<runtime::he::HeaanPlaintextWrapper> arg1_heaan =
            dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(arg1);
        shared_ptr<runtime::he::HeaanPlaintextWrapper> out_heaan =
            dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(out);

        if (arg0_heaan && arg1_heaan && out_heaan)
        {
            kernel::heaan::scalar_add(arg0_heaan, arg1_heaan, out_heaan, type, he_heaan_backend);
            out = dynamic_pointer_cast<runtime::he::HEPlaintext>(out_heaan);
        }
        else
        {
            throw ngraph_error(
                "Add backend is HEAAN, but arguments or outputs are not HeaanPlaintextWrapper.");
        }
    }
    else
    {
        throw ngraph_error("Add backend is neither SEAL nor HEAAN.");
    }
}

void runtime::he::kernel::scalar_add(const shared_ptr<runtime::he::HECiphertext>& arg0,
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
                kernel::seal::scalar_add(arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_seal);
            }
        }
        else
        {
            throw ngraph_error(
                "Add backend is SEAL, but arguments or outputs are not SealPlaintextWrapper");
        }
    }
    else if (auto he_heaan_backend =
                 dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(he_backend))
    {
        shared_ptr<runtime::he::HeaanCiphertextWrapper> arg0_heaan =
            dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(arg0);
        shared_ptr<runtime::he::HeaanPlaintextWrapper> arg1_heaan =
            dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(arg1);
        shared_ptr<runtime::he::HeaanCiphertextWrapper> out_heaan =
            dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(out);

        if (arg0_heaan && arg1_heaan && out_heaan)
        {
            auto zero = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(
                he_heaan_backend->get_valued_plaintext(0, type));

            if (arg1_heaan->m_plaintexts == zero->m_plaintexts)
            {
                out = arg0;
            }
            else
            {
                kernel::heaan::scalar_add(
                    arg0_heaan, arg1_heaan, out_heaan, type, he_heaan_backend);
                out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_heaan);
            }
        }
        else
        {
            throw ngraph_error(
                "Add backend is HEAAN, but arguments or outputs are not HeaanPlaintextWrapper.");
        }
    }
    else
    {
        throw ngraph_error("Add backend is neither SEAL nor HEAAN.");
    }
}

void runtime::he::kernel::scalar_add(const shared_ptr<runtime::he::HEPlaintext>& arg0,
                                     const shared_ptr<runtime::he::HECiphertext>& arg1,
                                     shared_ptr<runtime::he::HECiphertext>& out,
                                     const element::Type& type,
                                     const shared_ptr<runtime::he::HEBackend>& he_backend)
{
    scalar_add(arg1, arg0, out, type, he_backend);
}
