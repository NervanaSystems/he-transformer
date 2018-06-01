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

vector<tuple<vector<shared_ptr<ngraph::runtime::TensorView>>,vector<shared_ptr<ngraph::runtime::TensorView>>>> generate_tensors(
    const vector<tuple<element::Type, Shape>>& output, const vector<tuple<element::Type, Shape>>& input,
    shared_ptr<ngraph::runtime::he::HEBackend> backend)
{
    using ret_tuple_type = tuple<vector<shared_ptr<ngraph::runtime::TensorView>>,vector<shared_ptr<ngraph::runtime::TensorView>>>;
    auto he_backend = static_pointer_cast<ngraph::runtime::he::he_heaan::HEHeaanBackend>(backend);

    vector<tuple<vector<shared_ptr<ngraph::runtime::TensorView>>,vector<shared_ptr<ngraph::runtime::TensorView>>>> ret;

    auto cipher_cipher = [&output, &input, &he_backend] ()
    {
        vector<shared_ptr<ngraph::runtime::TensorView>> result;
        for (auto elem : output)
        {
            auto output_tensor = he_backend->create_tensor(get<0>(elem), get<1>(elem));
            result.push_back(output_tensor);
        }
        vector<shared_ptr<ngraph::runtime::TensorView>> argument;
        for (auto elem : input)
        {
            auto input_tensor = he_backend->create_tensor(get<0>(elem), get<1>(elem));
            argument.push_back(input_tensor);
        }
        return make_tuple(result, argument);
    };
    auto plain_plain = [&output, &input, &he_backend] ()
    {
        vector<shared_ptr<ngraph::runtime::TensorView>> result;
        for (auto elem : output)
        {
            auto output_tensor = he_backend->create_plain_tensor(get<0>(elem), get<1>(elem));
            result.push_back(output_tensor);
        }
        vector<shared_ptr<ngraph::runtime::TensorView>> argument;
        for (auto elem : input)
        {
            auto input_tensor = he_backend->create_plain_tensor(get<0>(elem), get<1>(elem));
            argument.push_back(input_tensor);
        }
        return make_tuple(result, argument);
    };
    auto alternate_cipher = [&output, &input, &he_backend] (size_t mod)
    {
        vector<shared_ptr<ngraph::runtime::TensorView>> result;
        for (auto elem : output)
        {
            auto output_tensor = he_backend->create_tensor(get<0>(elem), get<1>(elem));
            result.push_back(output_tensor);
        }
        vector<shared_ptr<ngraph::runtime::TensorView>> argument;
        for(size_t i = 0; i < input.size(); ++i)
        {
            auto elem = input[i];
            if (i % 2 == mod)
            {
                auto input_tensor = he_backend->create_plain_tensor(get<0>(elem), get<1>(elem));
                argument.push_back(input_tensor);
            }
            else
            {
                auto input_tensor = he_backend->create_tensor(get<0>(elem), get<1>(elem));
                argument.push_back(input_tensor);
            }
        }
        return make_tuple(result, argument);
    };
    auto plain_cipher_cipher = [&output, &input, &he_backend, &alternate_cipher] ()
    {
        return alternate_cipher(0);
    };

    auto cipher_plain_cipher= [&output, &input, &he_backend, &alternate_cipher] ()
    {
        return alternate_cipher(1);
    };

    ret.push_back(cipher_cipher());
    ret.push_back(plain_plain());
    if (input.size() >= 2)
    {
        ret.push_back(plain_cipher_cipher());
        ret.push_back(cipher_plain_cipher());
    }

    return ret;
}
