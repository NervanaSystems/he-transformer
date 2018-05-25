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

#include <fstream>
#include <iostream>
#include <sstream>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"

#include "he_backend.hpp"
#include "he_heaan_backend.hpp"
#include "he_seal_backend.hpp"
#include "test_util.hpp"

using namespace std;
using namespace ngraph;

void TestHEBackend::TearDown()
{
    cout << "Tearing down" << endl;
    // m_he_backend->clear_function_instance(); // TODO: add
    cout << "Tore down" << endl;
}

void TestHEBackend::SetUp()
{
    m_he_seal_backend = static_pointer_cast<runtime::he::he_seal::HESealBackend>(
        runtime::Backend::create("HE_SEAL"));
    m_he_heaan_backend = static_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(
        runtime::Backend::create("HE_HEAAN"));
}

shared_ptr<ngraph::runtime::he::he_seal::HESealBackend> TestHEBackend::m_he_seal_backend = nullptr;
shared_ptr<ngraph::runtime::he::he_heaan::HEHeaanBackend> TestHEBackend::m_he_heaan_backend =
    nullptr;

vector<float> read_constant(const string filename)
{
    string data = file_util::read_file_to_string(filename);
    istringstream iss(data);

    vector<string> constants;
    copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(constants));

    vector<float> res;
    for (const string& constant : constants)
    {
        res.push_back(atof(constant.c_str()));
    }
    return res;
}
