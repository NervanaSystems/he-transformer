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
#include "kernel/multiply.hpp"
#include "kernel/seal/multiply_seal.hpp"
#include "kernel/heaan/multiply_heaan.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/seal.h"
#include "util.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::multiply(const vector<shared_ptr<he::HECiphertext>>& arg0,
                                   const vector<shared_ptr<he::HECiphertext>>& arg1,
                                   vector<shared_ptr<he::HECiphertext>>& out,
                                   const element::Type& type,
                                   shared_ptr<HEBackend> he_backend,
                                   size_t count)
{
 #pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_multiply(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::scalar_multiply(const shared_ptr<he::HECiphertext>& arg0,
                                          const shared_ptr<he::HECiphertext>& arg1,
                                          shared_ptr<he::HECiphertext>& out,
                                          const element::Type& type,
                                          shared_ptr<HEBackend> he_backend)
{
    if(auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(he_backend))
    {
        shared_ptr<he::SealCiphertextWrapper> arg0_seal = dynamic_pointer_cast<he::SealCiphertextWrapper>(arg0);
        shared_ptr<he::SealCiphertextWrapper> arg1_seal = dynamic_pointer_cast<he::SealCiphertextWrapper>(arg1);
        shared_ptr<he::SealCiphertextWrapper> out_seal = dynamic_pointer_cast<he::SealCiphertextWrapper>(out);

        if (arg0_seal && arg1_seal && out_seal)
        {
            kernel::seal::scalar_multiply(arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
            out = dynamic_pointer_cast<he::HECiphertext>(out_seal);
        }
        else
        {
            throw ngraph_error(
                    "Multiply backend is seal, but arguments or outputs are not SealCiphertextWrapper");
        }
    }
    else if(auto he_heaan_backend = dynamic_pointer_cast<he_heaan::HEHeaanBackend>(he_backend))
    {
        shared_ptr<he::HeaanCiphertextWrapper> arg0_heaan = dynamic_pointer_cast<he::HeaanCiphertextWrapper>(arg0);
        shared_ptr<he::HeaanCiphertextWrapper> arg1_heaan = dynamic_pointer_cast<he::HeaanCiphertextWrapper>(arg1);
        shared_ptr<he::HeaanCiphertextWrapper> out_heaan = dynamic_pointer_cast<he::HeaanCiphertextWrapper>(out);

        if (arg0_heaan && arg1_heaan && out_heaan)
        {
            kernel::heaan::scalar_multiply(arg0_heaan, arg1_heaan, out_heaan, type, he_heaan_backend);
            out = dynamic_pointer_cast<he::HECiphertext>(out_heaan);
        }
        else
        {
            throw ngraph_error(
                    "Multiply backend is heaan, but arguments or outputs are not HeaanCiphertextWrapper");
        }
    }
    else
    {
        throw ngraph_error("Multiply backend is neither heaan nor seal.");
    }
}

void runtime::he::kernel::multiply(const vector<shared_ptr<he::HECiphertext>>& arg0,
                                   const vector<shared_ptr<he::HEPlaintext>>& arg1,
                                   vector<shared_ptr<he::HECiphertext>>& out,
                                   const element::Type& type,
                                   shared_ptr<HEBackend> he_backend,
                                   size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_multiply(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::scalar_multiply(const shared_ptr<he::HECiphertext>& arg0,
                                          const shared_ptr<he::HEPlaintext>& arg1,
                                          shared_ptr<he::HECiphertext>& out,
                                          const element::Type& type,
                                          shared_ptr<HEBackend> he_backend)
{
    if(auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(he_backend))
    {
        shared_ptr<he::SealCiphertextWrapper> arg0_seal = dynamic_pointer_cast<he::SealCiphertextWrapper>(arg0);
        shared_ptr<he::SealPlaintextWrapper> arg1_seal = dynamic_pointer_cast<he::SealPlaintextWrapper>(arg1);
        shared_ptr<he::SealCiphertextWrapper> out_seal = dynamic_pointer_cast<he::SealCiphertextWrapper>(out);

        if (arg0_seal && arg1_seal && out_seal)
        {
            kernel::seal::scalar_multiply(arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
            out = dynamic_pointer_cast<he::HECiphertext>(out_seal);
        }
        else
        {
            throw ngraph_error(
                    "Multiply backend is seal, but arguments or outputs are not SealCiphertextWrapper");
        }
    }
    else if(auto he_heaan_backend = dynamic_pointer_cast<he_heaan::HEHeaanBackend>(he_backend))
    {
        shared_ptr<he::HeaanCiphertextWrapper> arg0_heaan = dynamic_pointer_cast<he::HeaanCiphertextWrapper>(arg0);
        shared_ptr<he::HeaanPlaintextWrapper> arg1_heaan = dynamic_pointer_cast<he::HeaanPlaintextWrapper>(arg1);
        shared_ptr<he::HeaanCiphertextWrapper> out_heaan = dynamic_pointer_cast<he::HeaanCiphertextWrapper>(out);

        if (arg0_heaan && arg1_heaan && out_heaan)
        {
            kernel::heaan::scalar_multiply(arg0_heaan, arg1_heaan, out_heaan, type, he_heaan_backend);
            out = dynamic_pointer_cast<he::HECiphertext>(out_heaan);
        }
        else
        {
            throw ngraph_error(
                    "Multiply backend is heaan, but arguments or outputs are not HeaanCiphertextWrapper");
        }
    }
    else
    {
        throw ngraph_error("Multiply backend is neither heaan nor seal.");
    }
}

void runtime::he::kernel::multiply(const vector<shared_ptr<he::HEPlaintext>>& arg0,
                                   const vector<shared_ptr<he::HECiphertext>>& arg1,
                                   vector<shared_ptr<he::HECiphertext>>& out,
                                   const element::Type& type,
                                   shared_ptr<HEBackend> he_backend,
                                   size_t count)
{
    multiply(arg1, arg0, out, type, he_backend, count);
}

void runtime::he::kernel::scalar_multiply(const shared_ptr<he::HEPlaintext>& arg0,
                                          const shared_ptr<he::HECiphertext>& arg1,
                                          shared_ptr<he::HECiphertext>& out,
                                          const element::Type& type,
                                          shared_ptr<HEBackend> he_backend)
{
    scalar_multiply(arg1, arg0, out, type, he_backend);
}

void runtime::he::kernel::multiply(const vector<shared_ptr<he::HEPlaintext>>& arg0,
                                   const vector<shared_ptr<he::HEPlaintext>>& arg1,
                                   vector<shared_ptr<he::HEPlaintext>>& out,
                                   const element::Type& type,
                                   shared_ptr<HEBackend> he_backend,
                                   size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_multiply(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::scalar_multiply(const shared_ptr<he::HEPlaintext>& arg0,
                                          const shared_ptr<he::HEPlaintext>& arg1,
                                          shared_ptr<he::HEPlaintext>& out,
                                          const element::Type& type,
                                          shared_ptr<HEBackend> he_backend)
{
    if(auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(he_backend))
    {
        shared_ptr<he::SealPlaintextWrapper> arg0_seal = dynamic_pointer_cast<he::SealPlaintextWrapper>(arg0);
        shared_ptr<he::SealPlaintextWrapper> arg1_seal = dynamic_pointer_cast<he::SealPlaintextWrapper>(arg1);
        shared_ptr<he::SealPlaintextWrapper> out_seal = dynamic_pointer_cast<he::SealPlaintextWrapper>(out);

        if (arg0_seal && arg1_seal && out_seal)
        {
            kernel::seal::scalar_multiply(arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
            out = dynamic_pointer_cast<he::HEPlaintext>(out_seal);
        }
        else
        {
            throw ngraph_error(
                    "Multiply backend is seal, but arguments or outputs are not SealPlaintextWrapper");
        }
    }
    else if(auto he_heaan_backend = dynamic_pointer_cast<he_heaan::HEHeaanBackend>(he_backend))
    {
        shared_ptr<he::HeaanPlaintextWrapper> arg0_heaan = dynamic_pointer_cast<he::HeaanPlaintextWrapper>(arg0);
        shared_ptr<he::HeaanPlaintextWrapper> arg1_heaan = dynamic_pointer_cast<he::HeaanPlaintextWrapper>(arg1);
        shared_ptr<he::HeaanPlaintextWrapper> out_heaan = dynamic_pointer_cast<he::HeaanPlaintextWrapper>(out);

        if (arg0_heaan && arg1_heaan && out_heaan)
        {
            kernel::heaan::scalar_multiply(arg0_heaan, arg1_heaan, out_heaan, type, he_heaan_backend);
            out = dynamic_pointer_cast<he::HEPlaintext>(out_heaan);
        }
        else
        {
            throw ngraph_error(
                    "Multiply backend is heaan, but arguments or outputs are not HeaanPlaintextWrapper");
        }
    }
    else
    {
        throw ngraph_error("Multiply backend is neither heaan nor seal.");
    }
}
