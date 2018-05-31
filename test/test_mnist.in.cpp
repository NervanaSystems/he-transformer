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

// #include "ngraph/file_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"

#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

#include "he_backend.hpp"
#include "he_heaan_backend.hpp"
#include "he_seal_backend.hpp"

#include "test_util.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, tf_mnist_softmax_5)
{
    auto backend = static_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(
        runtime::Backend::create("${BACKEND_NAME}"));

    NGRAPH_INFO << "Loaded backend";
    const string filename = "mnist_softmax_batch5";
    const string json_path = file_util::path_join(HE_SERIALIZED_ZOO, filename + ".js");
    const string json_string = file_util::read_file_to_string(json_path);
    shared_ptr<Function> f = deserialize(json_string);

    // Visualize model
    auto model_file_name = filename + string(".") + pass::VisualizeTree::get_file_ext();

    unordered_map<string, vector<float>> parms;
    parms["b"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "weights/b.txt"));
    parms["W"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "weights/W.txt"));

    parms["x"] = read_constant(file_util::path_join(HE_SERIALIZED_ZOO, "weights/x.txt"));

    unordered_map<string, Shape> parm_shapes;
    parm_shapes["W"] = Shape{784, 10};
    parm_shapes["b"] = Shape{10};
    parm_shapes["x"] = Shape{5, 784};

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

    NGRAPH_INFO << "calling function";
    backend->call(f, result_tvs, parameter_tvs);

    EXPECT_TRUE(test::all_close(
        vector<float>{0.624406, -8.56023,  1.06847,  4.81459,    -2.81721,  -0.251675, -6.91895,
                      10.1022,  -0.612656, 2.55105,  4.25543,    -0.750629, 10.6834,   4.89307,
                      -10.9504, 5.10364,   6.01537,  -14.0582,   3.42761,   -8.61934,  -5.25755,
                      6.30253,  1.80438,   0.71403,  -2.37156,   -0.347839, -0.12199,  0.109456,
                      0.678508, -1.50997,  11.7059,  -11.5038,   1.67892,   0.275615,  -6.33494,
                      3.26604,  1.46237,   0.229319, -0.0319397, -0.747563, -1.17459,  -6.39749,
                      0.336039, -2.4198,   6.42819,  -1.60604,   0.56934,   1.30156,   0.565424,
                      2.39734},
        read_vector<float>(result_tvs[0]),
        1e-4f));
}

NGRAPH_TEST(${BACKEND_NAME}, tf_mnist_softmax_quantized_1)
{
    auto backend = static_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(
        runtime::Backend::create("${BACKEND_NAME}"));

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
