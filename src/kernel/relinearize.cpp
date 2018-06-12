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

#include "kernel/relinearize.hpp"
#include "he_backend.hpp"
#include "he_heaan_backend.hpp"
#include "he_seal_backend.hpp"
#include "ngraph/type/element_type.hpp"
#include "kernel/seal/relinearize_seal.hpp"
#include "kernel/heaan/relinearize_heaan.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::relinearize(const vector<shared_ptr<runtime::he::HECiphertext>>& arg,
                                      vector<shared_ptr<runtime::he::HECiphertext>>& out,
                                      const shared_ptr<runtime::he::HEBackend>& he_backend,
                                      size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        relinearize(arg[i], out[i], he_backend);
    }
}

void runtime::he::kernel::relinearize(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg,
                                      vector<shared_ptr<runtime::he::HEPlaintext>>& out,
                                      const shared_ptr<runtime::he::HEBackend>& he_backend,
                                      size_t count)
{
// Relinearize op doesn't make sense for Plaintexts. Just pass along to the output
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        out[i] = arg[i];
    }
}

void runtime::he::kernel::relinearize(const shared_ptr<runtime::he::HECiphertext>& arg,
                                      shared_ptr<runtime::he::HECiphertext>& out,
                                      const shared_ptr<runtime::he::HEBackend>& he_backend)
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
            kernel::seal::scalar_relinearize(arg_seal, out_seal, he_seal_backend);
            out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_seal);
        }
        else
        {
            throw ngraph_error(
                "Relinearize backend is SEAL, but argument or output is not SealPlaintextWrapper.");
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
            kernel::heaan::scalar_relinearize(arg_heaan, out_heaan, he_heaan_backend);
            out = dynamic_pointer_cast<runtime::he::HECiphertext>(out_heaan);
        }
        else
        {
            throw ngraph_error(
                    "Relinearize backend is HEAAN, but argument or output is not SealPlaintextWrapper.");
        }
    }
    else
    {
        throw ngraph_error("Relinearize backend is neither SEAL nor heaan.");
    }
}
