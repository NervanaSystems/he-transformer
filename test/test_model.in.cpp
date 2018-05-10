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
#include "loader.hpp"
#include <unordered_map>

TEST_F(TestHEBackend, tf_ptb)
{
    auto backend = runtime::Backend::create("CPU");
    const string model_name = "ptb_rnn";
    const string json_path = file_util::path_join(HE_SERIALIZED_ZOO, model_name + ".js");
    const string json_string = file_util::read_file_to_string(json_path);
    shared_ptr<Function> f = deserialize(json_string);

    // Get parameters
    //string U = file_util::read_file_to_string("U.save");
    unordered_map<string, vector<float>> parms;

    parms["U"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "U.save"));
    parms["W"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "W.save"));
    parms["b"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "b.save"));
    parms["Wy"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "Wy.save"));
    parms["by"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "by.save"));
    parms["x"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "x.save"));

    const size_t num_hidden = 500;
    const size_t time_steps = 10;
    const size_t vocab_size = 50;

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
        auto parameter_cipher_tv = backend->create_tensor(type, shape);
		auto parameter_plain_tv = backend->create_plain_tensor(type, shape);

		bool data_input = false;

        if (shape == Shape(hidden_size, hidden_size))
		{
			copy_data(parameter_plain_tv, parms["U"]);
		}
		else if (shape == Shape(vocab_size, hidden_size))
		{
			copy_data(parameter_plain_tv, parms["W"]);
		}
		else if (shape == Shape(hidden_size, vocab_size))
		{
			copy_data(parameter_plain_tv, parms["Wy"]);
		}
		else if (shape == Shape(vocab_size))
		{
			copy_data(parameter_plain_tv, parms["by"]);
		}
		else if (shape == Shape(hidden_size))
		{
			copy_data(parameter_plain_tv, parms["b"]);
		}
		else if (shape == Shape(vocab_size, time_steps))
		{
			copy_data(parameter_cipher_tv, parms["x"]);
			data_input = true;
		}
		else
		{
			throw ngraph_error("Unknown shape");
		}

        NGRAPH_INFO << "created tensor of " << shape_size(shape) << " elements";

if (data_input)
{

}
else {
	parameter_tvs.push_back(parameter_plain_tv);
}
        // copy_data(parameter_tv, vector<float>(shape_size(shape)));
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

    bool verbose = false;
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
