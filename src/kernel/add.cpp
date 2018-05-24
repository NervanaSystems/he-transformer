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
#include "he_heaan_backend.hpp"
#include "kernel/add.hpp"
#include "kernel/heaan/add_heaan.hpp"
#include "kernel/seal/add_seal.hpp"
#include "ngraph/type/element_type.hpp"
#include "util.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::add(const vector<shared_ptr<he::HECiphertext>>& arg0,
                              const vector<shared_ptr<he::HECiphertext>>& arg1,
                              vector<shared_ptr<he::HECiphertext>>& out,
                              shared_ptr<HEBackend> he_backend,
                              size_t count)
{
    if(auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(he_backend))
    {
        vector<shared_ptr<he::SealCiphertextWrapper>> arg0_seal(arg0.size());
        vector<shared_ptr<he::SealCiphertextWrapper>> arg1_seal(arg1.size());
        vector<shared_ptr<he::SealCiphertextWrapper>> out_seal(out.size());

        if (cast_vector(arg0_seal, arg0) && cast_vector(arg1_seal, arg1) && cast_vector(out_seal , out))
        {
            kernel::seal::add(arg0_seal, arg1_seal, out_seal, he_seal_backend, count);
            cast_vector(out, out_seal);
        }
        else
        {
            throw ngraph_error(
                    "Add backend is seal, but arguments or outputs are not SealCiphertextWrapper");
        }
    }
    else if(auto he_heaan_backend = dynamic_pointer_cast<he_heaan::HEHeaanBackend>(he_backend))
    {
        vector<shared_ptr<he::HeaanCiphertextWrapper>> arg0_heaan(arg0.size());
        vector<shared_ptr<he::HeaanCiphertextWrapper>> arg1_heaan(arg1.size());
        vector<shared_ptr<he::HeaanCiphertextWrapper>> out_heaan(out.size());

        if (cast_vector(arg0_heaan, arg0) && cast_vector(arg1_heaan, arg1) && cast_vector(out_heaan , out))
        {
            kernel::heaan::add(arg0_heaan, arg1_heaan, out_heaan, he_heaan_backend, count);
            cast_vector(out, out_heaan);
        }
        else
        {
            throw ngraph_error(
                    "Add backend is heaan, but arguments or outputs are not HeaanCiphertextWrapper");
        }
    }
    else
    {
        throw ngraph_error("Add backend is neither heaan nor seal.");
    }
}

void runtime::he::kernel::add(const vector<shared_ptr<he::HECiphertext>>& arg0,
                              const vector<shared_ptr<he::HEPlaintext>>& arg1,
                              vector<shared_ptr<he::HECiphertext>>& out,
                              shared_ptr<HEBackend> he_backend,
                              size_t count)
{
    if(auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(he_backend))
    {
        vector<shared_ptr<he::SealCiphertextWrapper>> arg0_seal(arg0.size());
        vector<shared_ptr<he::SealPlaintextWrapper>> arg1_seal(arg1.size());
        vector<shared_ptr<he::SealCiphertextWrapper>> out_seal(out.size());

        if (cast_vector(arg0_seal, arg0) && cast_vector(arg1_seal, arg1) && cast_vector(out_seal , out))
        {
            kernel::seal::add(arg0_seal, arg1_seal, out_seal, he_seal_backend, count);
            cast_vector(out, out_seal);
        }
        else
        {
            throw ngraph_error(
                    "Add backend is seal, but arguments or outputs are not SealCiphertextWrapper");
        }
    }
    else if(auto he_heaan_backend = dynamic_pointer_cast<he_heaan::HEHeaanBackend>(he_backend))
    {
        vector<shared_ptr<he::HeaanCiphertextWrapper>> arg0_heaan(arg0.size());
        vector<shared_ptr<he::HeaanPlaintextWrapper>> arg1_heaan(arg1.size());
        vector<shared_ptr<he::HeaanCiphertextWrapper>> out_heaan(out.size());

        if (cast_vector(arg0_heaan, arg0) && cast_vector(arg1_heaan, arg1) && cast_vector(out_heaan , out))
        {
            kernel::heaan::add(arg0_heaan, arg1_heaan, out_heaan, he_heaan_backend, count);
            cast_vector(out, out_heaan);
        }
        else
        {
            throw ngraph_error(
                    "Add backend is heaan, but arguments or outputs are not HeaanCiphertextWrapper");
        }
    }
    else
    {
        throw ngraph_error("Add backend is neither heaan nor seal.");
    }
}

void runtime::he::kernel::add(const vector<shared_ptr<he::HEPlaintext>>& arg0,
                              const vector<shared_ptr<he::HECiphertext>>& arg1,
                              vector<shared_ptr<he::HECiphertext>>& out,
                              shared_ptr<HEBackend> he_backend,
                              size_t count)
{
    add(arg1, arg0, out, he_backend, count);
}

void runtime::he::kernel::add(const vector<shared_ptr<he::HEPlaintext>>& arg0,
                              const vector<shared_ptr<he::HEPlaintext>>& arg1,
                              vector<shared_ptr<he::HEPlaintext>>& out,
                              const element::Type& type,
                              shared_ptr<HEBackend> he_backend,
                              size_t count)
{
    if(auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(he_backend))
    {
        vector<shared_ptr<he::SealPlaintextWrapper>> arg0_seal(arg0.size());
        vector<shared_ptr<he::SealPlaintextWrapper>> arg1_seal(arg1.size());
        vector<shared_ptr<he::SealPlaintextWrapper>> out_seal(out.size());

        if (cast_vector(arg0_seal, arg0) && cast_vector(arg1_seal, arg1) && cast_vector(out_seal , out))
        {
            kernel::seal::add(arg0_seal, arg1_seal, out_seal, type, he_seal_backend, count);
            cast_vector(out, out_seal);
        }
        else
        {
            throw ngraph_error(
                    "Add backend is seal, but arguments or outputs are not SealPlaintextWrapper");
        }
    }
    else if(auto he_heaan_backend = dynamic_pointer_cast<he_heaan::HEHeaanBackend>(he_backend))
    {
        vector<shared_ptr<he::HeaanPlaintextWrapper>> arg0_heaan(arg0.size());
        vector<shared_ptr<he::HeaanPlaintextWrapper>> arg1_heaan(arg1.size());
        vector<shared_ptr<he::HeaanPlaintextWrapper>> out_heaan(out.size());

        if (cast_vector(arg0_heaan, arg0) && cast_vector(arg1_heaan, arg1) && cast_vector(out_heaan , out))
        {
            kernel::heaan::add(arg0_heaan, arg1_heaan, out_heaan, type, he_heaan_backend, count);
            cast_vector(out, out_heaan);
        }
        else
        {
            throw ngraph_error(
                    "Add backend is heaan, but arguments or outputs are not HeaanPlaintextWrapper");
        }
    }
    else
    {
        throw ngraph_error("Add backend is neither heaan nor seal.");
    }
}

void runtime::he::kernel::scalar_add(const shared_ptr<he::HECiphertext>& arg0,
                                     const shared_ptr<he::HECiphertext>& arg1,
                                     shared_ptr<he::HECiphertext>& out,
                                     shared_ptr<HEBackend> he_backend)
{
    if(auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(he_backend))
    {
        shared_ptr<he::SealCiphertextWrapper> arg0_seal = dynamic_pointer_cast<he::SealCiphertextWrapper>(arg0);;
        shared_ptr<he::SealCiphertextWrapper> arg1_seal = dynamic_pointer_cast<he::SealCiphertextWrapper>(arg1_seal);;
        shared_ptr<he::SealCiphertextWrapper> out_seal = dynamic_pointer_cast<he::SealCiphertextWrapper>(out_seal);;

        if (arg0_seal && arg1_seal && out_seal)
        {
            kernel::seal::scalar_add(arg0_seal, arg1_seal, out_seal, he_seal_backend);
            out = dynamic_pointer_cast<he::HECiphertext>(out_seal);
        }
        else
        {
            throw ngraph_error(
                    "Add backend is seal, but arguments or outputs are not SealCiphertextWrapper");
        }
    }
    if(auto he_heaan_backend = dynamic_pointer_cast<he_heaan::HEHeaanBackend>(he_backend))
    {
        shared_ptr<he::HeaanCiphertextWrapper> arg0_heaan = dynamic_pointer_cast<he::HeaanCiphertextWrapper>(arg0);;
        shared_ptr<he::HeaanCiphertextWrapper> arg1_heaan = dynamic_pointer_cast<he::HeaanCiphertextWrapper>(arg1_heaan);;
        shared_ptr<he::HeaanCiphertextWrapper> out_heaan = dynamic_pointer_cast<he::HeaanCiphertextWrapper>(out_heaan);;

        if (arg0_heaan && arg1_heaan && out_heaan)
        {
            kernel::heaan::scalar_add(arg0_heaan, arg1_heaan, out_heaan, he_heaan_backend);
            out = dynamic_pointer_cast<he::HECiphertext>(out_heaan);
        }
        else
        {
            throw ngraph_error(
                    "Add backend is heaan, but arguments or outputs are not HeaanCiphertextWrapper");
        }
    }
    else
    {
        throw ngraph_error("Add backend is neither heaan nor seal.");
    }
}

void runtime::he::kernel::scalar_add(const shared_ptr<he::HEPlaintext>& arg0,
                                     const shared_ptr<he::HEPlaintext>& arg1,
                                     shared_ptr<he::HEPlaintext>& out,
                                     const element::Type& type,
                                     shared_ptr<HEBackend> he_backend)
{
    if(auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(he_backend))
    {
        shared_ptr<he::SealPlaintextWrapper> arg0_seal = dynamic_pointer_cast<he::SealPlaintextWrapper>(arg0);;
        shared_ptr<he::SealPlaintextWrapper> arg1_seal = dynamic_pointer_cast<he::SealPlaintextWrapper>(arg1_seal);;
        shared_ptr<he::SealPlaintextWrapper> out_seal = dynamic_pointer_cast<he::SealPlaintextWrapper>(out_seal);;

        if (arg0_seal && arg1_seal && out_seal)
        {
            kernel::seal::scalar_add(arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
            out = dynamic_pointer_cast<he::HEPlaintext>(out_seal);
        }
        else
        {
            throw ngraph_error(
                    "Add backend is seal, but arguments or outputs are not SealPlaintextWrapper");
        }
    }
    if(auto he_heaan_backend = dynamic_pointer_cast<he_heaan::HEHeaanBackend>(he_backend))
    {
        shared_ptr<he::HeaanPlaintextWrapper> arg0_heaan = dynamic_pointer_cast<he::HeaanPlaintextWrapper>(arg0);;
        shared_ptr<he::HeaanPlaintextWrapper> arg1_heaan = dynamic_pointer_cast<he::HeaanPlaintextWrapper>(arg1_heaan);;
        shared_ptr<he::HeaanPlaintextWrapper> out_heaan = dynamic_pointer_cast<he::HeaanPlaintextWrapper>(out_heaan);;

        if (arg0_heaan && arg1_heaan && out_heaan)
        {
            kernel::heaan::scalar_add(arg0_heaan, arg1_heaan, out_heaan, type, he_heaan_backend);
            out = dynamic_pointer_cast<he::HEPlaintext>(out_heaan);
        }
        else
        {
            throw ngraph_error(
                    "Add backend is heaan, but arguments or outputs are not HeaanPlaintextWrapper");
        }
    }
    else
    {
        throw ngraph_error("Add backend is neither heaan nor seal.");
    }
}
