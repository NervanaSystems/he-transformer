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
#include "kernel/heaan/negate_heaan.hpp"
#include "kernel/negate.hpp"
#include "kernel/seal/negate_seal.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::negate(const vector<shared_ptr<runtime::he::HECiphertext>>& arg,
                                 vector<shared_ptr<runtime::he::HECiphertext>>& out,
                                 const element::Type& type,
                                 shared_ptr<runtime::he::HEBackend> he_backend,
                                 size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_negate(arg[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::negate(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg,
                                 vector<shared_ptr<runtime::he::HEPlaintext>>& out,
                                 const element::Type& type,
                                 shared_ptr<runtime::he::HEBackend> he_backend,
                                 size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_negate(arg[i], out[i], type, he_backend);
    }
}

void runtime::he::kernel::scalar_negate(const shared_ptr<runtime::he::HECiphertext>& arg,
                                        shared_ptr<runtime::he::HECiphertext>& out,
                                        const element::Type& type,
                                        shared_ptr<runtime::he::HEBackend> he_backend)
{
    if (auto he_seal_backend =
            dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(he_backend))
    {
        shared_ptr<runtime::he::SealCiphertextWrapper> arg_seal =
            dynamic_pointer_cast<runtime::he::SealCiphertextWrapper>(arg);
        shared_ptr<runtime::he::SealCiphertextWrapper> out_seal =
            dynamic_pointer_cast<runtime::he::SealCiphertextWrapper>(out);

        if (arg_seal && out_seal)
        {
            kernel::seal::scalar_negate(arg_seal, out_seal, type, he_seal_backend);
            out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_seal);
        }
        else
        {
            throw ngraph_error(
                "negate backend is seal, but arguments or outputs are not SealCiphertextWrapper");
        }
    }
    else if (auto he_heaan_backend =
                 dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(he_backend))
    {
        shared_ptr<runtime::he::HeaanCiphertextWrapper> arg_heaan =
            dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(arg);
        shared_ptr<runtime::he::HeaanCiphertextWrapper> out_heaan =
            dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(out);

        if (arg_heaan && out_heaan)
        {
            kernel::heaan::scalar_negate(arg_heaan, out_heaan, type, he_heaan_backend);
            out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_heaan);
        }
        else
        {
            throw ngraph_error(
                "negate backend is heaan, but arguments or outputs are not HeaanCiphertextWrapper");
        }
    }
    else
    {
        throw ngraph_error("negate backend is neither seal nor hean.");
    }
}

void runtime::he::kernel::scalar_negate(const shared_ptr<runtime::he::HEPlaintext>& arg,
                                        shared_ptr<runtime::he::HEPlaintext>& out,
                                        const element::Type& type,
                                        shared_ptr<runtime::he::HEBackend> he_backend)
{
    if (auto he_seal_backend =
            dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(he_backend))
    {
        shared_ptr<runtime::he::SealPlaintextWrapper> arg_seal =
            dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(arg);
        shared_ptr<runtime::he::SealPlaintextWrapper> out_seal =
            dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(out);

        if (arg_seal && out_seal)
        {
            kernel::seal::scalar_negate(arg_seal, out_seal, type, he_seal_backend);
            out = dynamic_pointer_cast<runtime::he::HEPlaintext>(out_seal);
        }
        else
        {
            throw ngraph_error(
                "negate backend is seal, but arguments or outputs are not SealPlaintextWrapper");
        }
    }
    else if (auto he_heaan_backend =
                 dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(he_backend))
    {
        shared_ptr<runtime::he::HeaanPlaintextWrapper> arg_heaan =
            dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(arg);
        shared_ptr<runtime::he::HeaanPlaintextWrapper> out_heaan =
            dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(out);

        if (arg_heaan && out_heaan)
        {
            kernel::heaan::scalar_negate(arg_heaan, out_heaan, type, he_heaan_backend);
            out = dynamic_pointer_cast<runtime::he::HEPlaintext>(out_heaan);
        }
        else
        {
            throw ngraph_error(
                "negate backend is heaan, but arguments or outputs are not HeaanPlaintextWrapper");
        }
    }
    else
    {
        throw ngraph_error("negate backend is neither seal nor hean.");
    }
}
