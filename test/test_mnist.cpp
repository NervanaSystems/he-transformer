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

#include "ngraph/file_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"

#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

#include "he_backend.hpp"
#include "he_heaan_backend.hpp"
#include "he_seal_backend.hpp"
#include "test_util.hpp"

using namespace std;
using namespace ngraph;

TEST_F(TestHEBackend, tf_mnist_deep_1)
{ // TODO: generalize to multiple backends
    shared_ptr<runtime::he::he_heaan::HEHeaanBackend> backend = dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>( runtime::Backend::create("HE_HEAAN"));
    // shared_ptr<runtime::he::he_seal::HESealBackend> backend = dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(runtime::Backend::create("HE_SEAL"));
    //
    const string backend_type = "HE_HEAAN";
    // const string backend_type = "INTERPRETER";
    // auto backend = runtime::Backend::create("INTERPRETER");
    NGRAPH_INFO << "Loaded backend";
    const string filename = "mnist_deep_simplified_batch_2";
    const string json_path = file_util::path_join(HE_SERIALIZED_ZOO, filename + ".js");
    const string json_string = file_util::read_file_to_string(json_path);
    shared_ptr<Function> f = deserialize(json_string);

    // Visualize model
    auto model_file_name = filename + string(".") + pass::VisualizeTree::get_file_ext();

    unordered_map<string, vector<float>> parms;
    parms["conv1_b"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "weights/conv1_b_conv1:0.txt"));
    parms["conv1_W"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "weights/W_conv1:0.txt"));

    parms["conv2_b"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "weights/conv2_b_conv2:0.txt"));
    parms["conv2_W"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "weights/W_conv2:0.txt"));

    parms["fc1_b"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "weights/fc1_b_fc1:0.txt"));
    parms["fc1_W"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "weights/W_fc1:0.txt"));

    parms["fc2_b"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "weights/fc2_b_fc2:0.txt"));
    parms["fc2_W"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "weights/W_fc2:0.txt"));

    parms["x"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "weights/x.txt"));

    unordered_map<string, Shape> parm_shapes;
    parm_shapes["conv1_W"] = Shape{5, 5, 1, 5};
    parm_shapes["conv1_b"] = Shape{5};
    parm_shapes["conv2_W"] = Shape{5, 5, 5, 11};
    parm_shapes["conv2_b"] = Shape{11};
    parm_shapes["fc1_W"] = Shape{539, 100};
    parm_shapes["fc1_b"] = Shape{100};
    parm_shapes["fc2_W"] = Shape{100, 10};
    parm_shapes["fc2_b"] = Shape{10};
    parm_shapes["x"] = Shape{2, 784};

    NGRAPH_INFO << "Deserialized graph";
    auto parameters = f->get_parameters();
    vector<shared_ptr<runtime::TensorView>> parameter_tvs;
    for (auto parameter : parameters)
    {
        auto& shape = parameter->get_shape();
        auto& type = parameter->get_element_type();
        auto parameter_cipher_tv = backend->create_tensor(type, shape);
        auto parameter_tv = backend->create_tensor(type, shape);
        auto parameter_plain_tv = backend->create_plain_tensor(type, shape);
        bool data_input = false;
        bool valid_shape = false;

        string parm;

        NGRAPH_INFO << join(shape, "x");

        for (auto const& it : parm_shapes)
        {
            if (it.second == shape)
            {
                valid_shape = true;
                parm = it.first;
                NGRAPH_INFO << "Adding " << parm;
                if (backend_type == "INTERPRETER")
                {
                    copy_data(parameter_tv, parms[parm]);
                    parameter_tvs.push_back(parameter_tv);
                }
                else if (backend_type == "HE_HEAAN")
                {
                    if (parm == "x")
                    {
                        NGRAPH_INFO << "Copying x size " << shape_size(shape);
                        copy_data(parameter_cipher_tv, parms[parm]);
                        parameter_tvs.push_back(parameter_cipher_tv);
                        NGRAPH_INFO << "Copied x";
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
                    throw ngraph_error("Unknown backend type");
                }
            }
        }
        if (!valid_shape)
        {
            NGRAPH_INFO << "Invalid shape" << join(shape, "x");
            throw ngraph_error("Invalid shape " + shape_size(shape));
        }
    }

    auto results = f->get_results();
    vector<shared_ptr<runtime::TensorView>> result_tvs;
    for (auto result : results)
    {
        auto& shape = result->get_shape();
        auto& type = result->get_element_type();
        result_tvs.push_back(backend->create_tensor(type, shape));
    }

    NGRAPH_INFO << "calling function on " << parameter_tvs.size() << " inputs";
    backend->call(f, result_tvs, parameter_tvs);
    NGRAPH_INFO << "called function";

    EXPECT_EQ((vector<float>{
0.0899273, 0.110965, 0.0533864, 0.0685337, 0.0973437, 0.191139, 0.0625438, 0.0817467, 0.125587, 0.0393772, 0.0550941, 0.105483, 0.0332782, 0.0753735, 0.102445, 0.213246, 0.0803351, 0.0786155, 0.116955, 0.0446246}),
            read_vector<float>(result_tvs[0]));
}

TEST_F(TestHEBackend, tf_mnist_const_1)
{ // TODO: generalize to multiple backends
    //shared_ptr<runtime::he::he_heaan::HEHeaanBackend> backend =
        dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(
                runtime::Backend::create("HE_HEAAN"));
    shared_ptr<runtime::he::he_seal::HESealBackend> backend = dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(runtime::Backend::create("HE_SEAL"));
    NGRAPH_INFO << "Loaded backend";
    const string json_path = file_util::path_join(HE_SERIALIZED_ZOO, "mnist_mlp_const_1_inputs.js");
    const string json_string = file_util::read_file_to_string(json_path);
    shared_ptr<Function> f = deserialize(json_string);

    NGRAPH_INFO << "Deserialized graph";
    auto parameters = f->get_parameters();
    vector<shared_ptr<runtime::TensorView>> parameter_tvs;
    for (auto parameter : parameters)
    {
        auto& shape = parameter->get_shape();
        auto& type = parameter->get_element_type();
        NGRAPH_INFO << "Creating tensor of " << shape_size(shape) << " elements";
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

    EXPECT_EQ((vector<float>{2173, 944, 1151, 1723, -1674, 569, -1985, 9776, -4997, -1903}),
            read_vector<float>(result_tvs[0]));
}

