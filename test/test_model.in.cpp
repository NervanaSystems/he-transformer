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

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include "ngraph/file_util.hpp"

TEST_F(TestHEBackend, tf_ptb)
{
    auto backend = runtime::Backend::create("CPU");
    const string model_name = "ptb_rnn";
    const string json_path = file_util::path_join(HE_SERIALIZED_ZOO, model_name + ".js");
    const string json_string = file_util::read_file_to_string(json_path);
    shared_ptr<Function> f = deserialize(json_string);

    // Visualize model
    auto model_file_name = model_name + string(".") + pass::VisualizeTree::get_file_ext();

    NGRAPH_INFO << "model file name " << model_file_name;
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>(model_file_name);
    pass_manager.run_passes(f);

    auto parameters = f->get_parameters();
    vector<shared_ptr<runtime::TensorView>> parameter_tvs;
    for (auto parameter : parameters)
    {
        auto& shape = parameter->get_shape();
        auto& type = parameter->get_element_type();
        auto parameter_tv = backend->create_tensor(type, shape);
        NGRAPH_INFO << "created tensor of " << shape_size(shape) << " elements";
        copy_data(parameter_tv, vector<float>(shape_size(shape)));
        parameter_tvs.push_back(parameter_tv);
    }

    auto results = f->get_results();
    vector<shared_ptr<runtime::TensorView>> result_tvs;
    for (auto result : results)
    {
        auto& shape = result->get_shape();
        auto& type = result->get_element_type();
        result_tvs.push_back(backend->create_tensor(type, shape));
    }

    NGRAPH_INFO << "calling function ";
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
