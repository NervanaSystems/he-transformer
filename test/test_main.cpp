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

// This is a special test file that contains `*.in.cpp` sources from multiple files
// On ngraph's main repo, add this file to the unit-test binary source list

// Gtest
#include "gtest/gtest.h"

// Ngraph
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"

// Ngraph test
#include "util/autodiff/numeric_compare.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

// HE Backend
#include "he_backend.hpp"
#include "test_main.hpp"

// Namespace
using namespace std;
using namespace ngraph;

// Common test class
void TestHEBackend::SetUp()
{
    m_he_backend = static_pointer_cast<runtime::he::HEBackend>(runtime::Backend::create("HE"));
}

// Source files
#include "test_basics.in.cpp"
#include "test_model.in.cpp"
