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

#include "kernel/seal/relinearize_seal.hpp"
#include "he_seal_backend.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::seal::scalar_relinearize(
    const shared_ptr<runtime::he::SealCiphertextWrapper>& arg,
    shared_ptr<runtime::he::SealCiphertextWrapper>& out,
    const shared_ptr<runtime::he::he_seal::HESealBackend> he_seal_backend)
{
    // It's safe to do inplace relinearize on the input since the un-relinearized result won't be
    // used by other ops. That is, this relinearize op is immediately after a multiply op, and the
    // relinearize op is the only op using the result from the multiply op
    he_seal_backend->get_evaluator()->relinearize(arg->m_ciphertext,
                                                  *(he_seal_backend->get_ev_key()));
    out = arg;
}
