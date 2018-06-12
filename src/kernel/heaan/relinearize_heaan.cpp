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

#include "kernel/heaan/relinearize_heaan.hpp"
#include "he_heaan_backend.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::heaan::scalar_relinearize(
    const shared_ptr<runtime::he::HeaanCiphertextWrapper>& arg,
    shared_ptr<runtime::he::HeaanCiphertextWrapper>& out,
    const shared_ptr<runtime::he::he_heaan::HEHeaanBackend> he_heaan_backend)
{
    // It's safe to do inplace relinearize on the input since the un-relinearized result won't be
    // used by other ops. That is, this relinearize op is immediately after a multiply op, and the
    // relinearize op is the only op using the result from the multiply op
    // HEAAN already performs relinearization after the multiply op, but we additionally need
    // to rescale.
    he_heaan_backend->get_scheme()->reScaleByAndEqual(arg->m_ciphertext,
                                                      he_heaan_backend->get_precision());
    out->m_ciphertext = arg->m_ciphertext;
}
