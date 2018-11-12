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

#include "ngraph/type/element_type.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/add_seal.hpp"
#include "seal/bfv/kernel/add_seal_bfv.hpp"
#include "seal/ckks/kernel/add_seal_ckks.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void he_seal::kernel::scalar_add(const shared_ptr<const he_seal::SealCiphertextWrapper>& arg0,
                                 const shared_ptr<const he_seal::SealCiphertextWrapper>& arg1,
                                 shared_ptr<he_seal::SealCiphertextWrapper>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealBackend* he_seal_backend)
{
    if (auto he_seal_ckks_backend =
            dynamic_cast<const he_seal::HESealCKKSBackend*>(he_seal_backend))
    {
        he_seal::ckks::kernel::scalar_add_ckks(arg0, arg1, out, element_type, he_seal_ckks_backend);
    }
    else if (auto he_seal_bfv_backend =
        dynamic_cast<const he_seal::HESealBFVBackend*>(he_seal_backend))
    {
        he_seal::bfv::kernel::scalar_add_bfv(arg0, arg1, out, element_type, he_seal_bfv_backend);
    }
    else
    {
        throw ngraph_error("HESealBackend is neither BFV nor CKKS");
    }
}

void he_seal::kernel::scalar_add(const shared_ptr<const he_seal::SealCiphertextWrapper>& arg0,
                                 const shared_ptr<const he_seal::SealPlaintextWrapper>& arg1,
                                 shared_ptr<he_seal::SealCiphertextWrapper>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealBackend* he_seal_backend)
{
    if (auto he_seal_ckks_backend =
            dynamic_cast<const he_seal::HESealCKKSBackend*>(he_seal_backend))
    {
        he_seal::ckks::kernel::scalar_add_ckks(arg0, arg1, out, element_type, he_seal_ckks_backend);
    }
    else if (auto he_seal_bfv_backend =
        dynamic_cast<const he_seal::HESealBFVBackend*>(he_seal_backend))
    {
        he_seal::bfv::kernel::scalar_add_bfv(arg0, arg1, out, element_type, he_seal_bfv_backend);
    }
    else
    {
        throw ngraph_error("HESealBackend is neither BFV nor CKKS");
    }
}

void he_seal::kernel::scalar_add(const shared_ptr<const he_seal::SealPlaintextWrapper>& arg0,
                                 const shared_ptr<const he_seal::SealCiphertextWrapper>& arg1,
                                 shared_ptr<he_seal::SealCiphertextWrapper>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealBackend* he_seal_backend)
{
    he_seal::kernel::scalar_add(arg1, arg0, out, element_type, he_seal_backend);
}

void he_seal::kernel::scalar_add(const shared_ptr<he_seal::SealPlaintextWrapper>& arg0,
                                 const shared_ptr<he_seal::SealPlaintextWrapper>& arg1,
                                 shared_ptr<he_seal::SealPlaintextWrapper>& out,
                                 const element::Type& element_type,
                                 const he_seal::HESealBackend* he_seal_backend)
{
    if (auto he_seal_ckks_backend =
            dynamic_cast<const he_seal::HESealCKKSBackend*>(he_seal_backend))
    {
        he_seal::ckks::kernel::scalar_add_ckks(arg0, arg1, out, element_type, he_seal_ckks_backend);
    }
    else if (auto he_seal_bfv_backend =
        dynamic_cast<const he_seal::HESealBFVBackend*>(he_seal_backend))
    {
        he_seal::bfv::kernel::scalar_add_bfv(arg0, arg1, out, element_type, he_seal_bfv_backend);
    }
    else
    {
        throw ngraph_error("HESealBackend is neither BFV nor CKKS");
    }
}