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

#include "kernel/heaan/square_heaan.hpp"
#include "he_heaan_backend.hpp"
#include "heaan_ciphertext_wrapper.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::heaan::scalar_square(
    const shared_ptr<runtime::he::HeaanCiphertextWrapper>& arg,
    shared_ptr<runtime::he::HeaanCiphertextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_heaan::HEHeaanBackend> he_heaan_backend)
{
    out->m_ciphertext = he_heaan_backend->get_scheme()->square(arg->m_ciphertext);
}
