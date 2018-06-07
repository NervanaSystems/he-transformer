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

#include "heaan_plaintext_wrapper.hpp"
#include "ngraph/except.hpp"

using namespace std;
using namespace ngraph;

runtime::he::HeaanPlaintextWrapper::HeaanPlaintextWrapper()
    : m_plaintexts()
{
}

runtime::he::HeaanPlaintextWrapper::HeaanPlaintextWrapper(std::vector<double>& plain)
{
    auto is_power_of_2 = [](size_t n) -> bool { return (n & (n - 1)) == 0; };

    if (is_power_of_2(plain.size()))
    {
        m_plaintexts = plain;
    }
    else
    {
        throw ngraph_error("Batching size must be power of two");
    }
}

runtime::he::HeaanPlaintextWrapper::~HeaanPlaintextWrapper()
{
}
