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

using namespace std;
using namespace ngraph;

void runtime::he::kernel::relinearize(const vector<shared_ptr<runtime::he::HECiphertext>>& arg,
                                      vector<shared_ptr<runtime::he::HECiphertext>>& out,
                                      const shared_ptr<runtime::he::HEBackend> he_backend,
                                      size_t count)
{
// It's safe to do inplace relinearize on the input since the un-relinearized result won't be
// used by other ops. That is, this relinearize op is immediately after a multiply op, and the
// relinearize op is the only op using the result from the multiply op
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        relinearize(arg[i], out[i], he_backend);
    }
}

void runtime::he::kernel::relinearize(const vector<shared_ptr<runtime::he::HEPlaintext>>& arg,
                                      vector<shared_ptr<runtime::he::HEPlaintext>>& out,
                                      const shared_ptr<runtime::he::HEBackend> he_backend,
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
                                      const shared_ptr<runtime::he::HEBackend> he_backend)
{
    if (auto he_seal_backend =
            dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(he_backend))
    {
        shared_ptr<runtime::he::SealCiphertextWrapper> arg_seal =
            dynamic_pointer_cast<runtime::he::SealCiphertextWrapper>(arg);
        if (arg_seal)
        {
            he_seal_backend->get_evaluator()->relinearize(arg_seal->m_ciphertext,
                                                          *(he_seal_backend->get_ev_key()));
            out = arg;
        }
        else
        {
            throw ngraph_error(
                "Relinearize backend is seal, but argument is not SealPlaintextWrapper.");
        }
    }
    else if (auto he_heaan_backend =
                 dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(he_backend))
    {
        out = arg;
    }
    else
    {
        throw ngraph_error("Relinearize backend is neither seal nor heaan.");
    }
}
