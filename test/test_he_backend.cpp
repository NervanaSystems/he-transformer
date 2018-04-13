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

#include <iostream>
#include <memory>
#include <vector>

// Gtest
#include "gtest/gtest.h"

// Ngraph
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"

// HE

// Ngraph test
#include "util/autodiff/numeric_compare.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

// HE test headers
#include "test_he_backend.hpp"
using namespace std;
using namespace ngraph;

// HE test src
void TestHEBackend::SetUp()
{
    m_he_manager = runtime::Manager::get("HE");
    m_he_backend = m_he_manager->allocate_backend();
}

static bool all_close_d(const vector<float>& a,
                        const vector<float>& b,
                        float rtol = float(1e-5),
                        float atol = float(1e-8))
{
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (abs(a[i] - b[i]) > atol + rtol * abs(b[i]))
        {
            return false;
        }
    }
    return true;
}
