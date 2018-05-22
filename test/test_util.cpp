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
#include "test_util.hpp"

using namespace std;
using namespace ngraph;

void TestHEBackend::TearDown()
{
    m_he_seal_backend->clear_function_instance();
}

shared_ptr<ngraph::runtime::he::he_seal::HESealBackend> TestHEBackend::m_he_seal_backend =
    static_pointer_cast<runtime::he::he_seal::HESealBackend>(runtime::Backend::create("HE_SEAL"));

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
