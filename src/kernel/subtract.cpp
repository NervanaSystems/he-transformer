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

#include "seal/he_seal_backend.hpp"
#include "kernel/subtract.hpp"
#include "kernel/add.hpp"
#include "kernel/negate.hpp"
#include "seal/kernel/subtract_seal.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void kernel::subtract(const vector<shared_ptr<HECiphertext>>& arg0,
                              const vector<shared_ptr<HECiphertext>>& arg1,
                              vector<shared_ptr<HECiphertext>>& out,
                              const element::Type& type,
                              const HEBackend* he_backend,
                              size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_subtract(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void kernel::subtract(const vector<shared_ptr<HECiphertext>>& arg0,
                              const vector<shared_ptr<HEPlaintext>>& arg1,
                              vector<shared_ptr<HECiphertext>>& out,
                              const element::Type& type,
                              const HEBackend* he_backend,
                              size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_subtract(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void kernel::subtract(const vector<shared_ptr<HEPlaintext>>& arg0,
                              const vector<shared_ptr<HECiphertext>>& arg1,
                              vector<shared_ptr<HECiphertext>>& out,
                              const element::Type& type,
                              const HEBackend* he_backend,
                              size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_subtract(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void kernel::subtract(const vector<shared_ptr<HEPlaintext>>& arg0,
                              const vector<shared_ptr<HEPlaintext>>& arg1,
                              vector<shared_ptr<HEPlaintext>>& out,
                              const element::Type& type,
                              const HEBackend* he_backend,
                              size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_subtract(arg0[i], arg1[i], out[i], type, he_backend);
    }
}

void kernel::scalar_subtract(const shared_ptr<HECiphertext>& arg0,
                                     const shared_ptr<HECiphertext>& arg1,
                                     shared_ptr<HECiphertext>& out,
                                     const element::Type& type,
                                     const HEBackend* he_backend)
{
    if (auto he_seal_backend =
            dynamic_cast<const he_seal::HESealBackend*>(he_backend))
    {
        shared_ptr<he_seal::SealCiphertextWrapper> arg0_seal =
            dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(arg0);
        shared_ptr<he_seal::SealCiphertextWrapper> arg1_seal =
            dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(arg1);
        shared_ptr<he_seal::SealCiphertextWrapper> out_seal =
            dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(out);

        if (arg0_seal && arg1_seal && out_seal)
        {
            he_seal::kernel::scalar_subtract(arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
            out = dynamic_pointer_cast<HECiphertext>(out_seal);
        }
        else
        {
            throw ngraph_error(
                "subtract backend is SEAL, but arguments or outputs are not SealCiphertextWrapper");
        }
    }
    else
    {
        throw ngraph_error("subtract backend is not SEAL.");
    }
}

void kernel::scalar_subtract(const shared_ptr<HEPlaintext>& arg0,
                                     const shared_ptr<HEPlaintext>& arg1,
                                     shared_ptr<HEPlaintext>& out,
                                     const element::Type& type,
                                     const HEBackend* he_backend)
{
    if (auto he_seal_backend =
            dynamic_cast<const he_seal::HESealBackend*>(he_backend))
    {
        shared_ptr<he_seal::SealPlaintextWrapper> arg0_seal =
            dynamic_pointer_cast<he_seal::SealPlaintextWrapper>(arg0);
        shared_ptr<he_seal::SealPlaintextWrapper> arg1_seal =
            dynamic_pointer_cast<he_seal::SealPlaintextWrapper>(arg1);
        shared_ptr<he_seal::SealPlaintextWrapper> out_seal =
            dynamic_pointer_cast<he_seal::SealPlaintextWrapper>(out);

        if (arg0_seal && arg1_seal && out_seal)
        {
            he_seal::kernel::scalar_subtract(arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
            out = dynamic_pointer_cast<HEPlaintext>(out_seal);
        }
        else
        {
            throw ngraph_error(
                "subtract backend is SEAL, but arguments or outputs are not SealPlaintextWrapper.:");
        }
    }
    else
    {
        throw ngraph_error("subtract backend is neither SEAL nor HEAAN.");
    }
}

void kernel::scalar_subtract(const shared_ptr<HECiphertext>& arg0,
                                     const shared_ptr<HEPlaintext>& arg1,
                                     shared_ptr<HECiphertext>& out,
                                     const element::Type& type,
                                     const HEBackend* he_backend)
{
    if (auto he_seal_backend =
            dynamic_cast<const he_seal::HESealBackend*>(he_backend))
    {
        shared_ptr<he_seal::SealCiphertextWrapper> arg0_seal =
            dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(arg0);
        shared_ptr<he_seal::SealPlaintextWrapper> arg1_seal =
            dynamic_pointer_cast<he_seal::SealPlaintextWrapper>(arg1);
        shared_ptr<he_seal::SealCiphertextWrapper> out_seal =
            dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(out);

        if (arg0_seal && arg1_seal && out_seal)
        {
            const string type_name = type.c_type_string();
            bool sub_zero = false;
            if (type_name == "int64_t") // TODO: make more general / templatize
            {
                int64_t x = 0;
                he_backend->decode((void*)(&x), arg1, type, 1);
                sub_zero = (x == 0);
            }
            else if (type_name == "float")
            {
                float x;
                he_backend->decode((void*)(&x), arg1, type, 1);
                sub_zero = (x == 0);
            }
            if (sub_zero)
            {
                NGRAPH_INFO << "Optimized subtract";
                out = arg0;
            }
            else
            {
                he_seal::kernel::scalar_subtract(arg0_seal, arg1_seal, out_seal, type, he_seal_backend);
                out = dynamic_pointer_cast<HECiphertext>(out_seal);
            }
        }
        else
        {
            throw ngraph_error(
                "subtract backend is SEAL, but arguments or outputs are not SealPlaintextWrapper");
        }
    }
    else
    {
        throw ngraph_error("subtract backend is neither SEAL nor HEAAN.");
    }
}

void kernel::scalar_subtract(const shared_ptr<HEPlaintext>& arg0,
                                     const shared_ptr<HECiphertext>& arg1,
                                     shared_ptr<HECiphertext>& out,
                                     const element::Type& type,
                                     const HEBackend* he_backend)
{
    auto neg_arg1 = arg1;
    scalar_negate(neg_arg1, neg_arg1, type, he_backend);
    scalar_add(arg0, neg_arg1, out, type, he_backend);
}
