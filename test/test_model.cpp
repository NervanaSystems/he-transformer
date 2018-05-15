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

#include <assert.h>

#include "ngraph/file_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

#include "he_backend.hpp"
#include "test_util.hpp"

using namespace std;
using namespace ngraph;

TEST_F(TestHEBackend, ptb_100)
{
    const size_t hidden_size = 100;
    const size_t time_steps = 4;
    const size_t vocab_size = 50;
    const size_t batch_size = 2;

    const string backend_type = "HE";
    auto backend = runtime::Backend::create(backend_type);
    const string model_name = "ptb_rnn100";
    const string json_path = file_util::path_join(HE_SERIALIZED_ZOO, model_name + ".js");
    const string json_string = file_util::read_file_to_string(json_path);
    shared_ptr<Function> f = deserialize(json_string);

    // Get parameters
    //string U = file_util::read_file_to_string("U.save");
    unordered_map<string, vector<float>> parms;

    parms["U"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "U100.save"));
    parms["W"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "W100.save"));
    parms["b"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "b100.save"));
    parms["Wy"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "Wy100.save"));
    parms["by"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "by100.save"));
    parms["x"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "x100.save"));

    auto parameters = f->get_parameters();
    vector<shared_ptr<runtime::TensorView>> parameter_tvs;
    for (auto parameter : parameters)
    {
        auto& shape = parameter->get_shape();
        auto& type = parameter->get_element_type();
        auto parameter_cipher_tv = backend->create_tensor(type, shape);
        auto parameter_tv = backend->create_tensor(type, shape);
        auto parameter_plain_tv = m_he_backend->create_plain_tensor(type, shape);
        bool data_input = false;

        string parm;

        if (shape == Shape{hidden_size, hidden_size})
        {
            parm = "U";
        }
        else if (shape == Shape{vocab_size, hidden_size})
        {
            parm = "W";
        }
        else if (shape == Shape{hidden_size, vocab_size})
        {
            parm = "Wy";
        }
        else if (shape == Shape{vocab_size})
        {
            parm = "by";
        }
        else if (shape == Shape{hidden_size})
        {
            parm = "b";
        }
        else if (shape == Shape{batch_size, time_steps})
        {
            parm = "x";
        }
        else
        {
            throw ngraph_error("Unknown shape");
        }

        if (backend_type == "CPU")
        {
            copy_data(parameter_tv, parms[parm]);
            parameter_tvs.push_back(parameter_tv);
        }
        else if (backend_type == "HE")
        {
            if (parm == "x")
            {
                copy_data(parameter_cipher_tv, parms[parm]);
                parameter_tvs.push_back(parameter_cipher_tv);
            }
            else
            {
                NGRAPH_INFO << "Creating plain tv size " << shape_size(shape);
                copy_data(parameter_plain_tv, parms[parm]);
                parameter_tvs.push_back(parameter_plain_tv);
            }
        }
        else
        {
            NGRAPH_INFO << "Unknown backend type";
        }

        NGRAPH_INFO << "created tensor of " << shape_size(shape) << " elements";
    }

    auto results = f->get_results();
    vector<shared_ptr<runtime::TensorView>> result_tvs;
    for (auto result : results)
    {
        auto& shape = result->get_shape();
        NGRAPH_INFO << "result shape " << join(shape, "x");
        auto& type = result->get_element_type();
        result_tvs.push_back(backend->create_tensor(type, shape));
    }

    NGRAPH_INFO << "calling function ";
    NGRAPH_INFO << "parameter_tvs.size " << parameter_tvs.size();

    // Visualize model
    if (auto he_backend = dynamic_pointer_cast<runtime::he::HEBackend>(backend))
    {
        auto model_file_name = model_name + string(".") + pass::VisualizeTree::get_file_ext();
        he_backend->visualize_function_after_pass(f, model_file_name);
    }

    backend->call(f, result_tvs, parameter_tvs);

    bool verbose = true;
    if (verbose)
    {
        cout << "Result" << endl;
        auto result = read_vector<float>(result_tvs[0]);
        for (auto elem : result)
        {
            cout << elem << endl;
        }
    }
}
