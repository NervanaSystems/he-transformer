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

TEST_F(TestHEBackend, dot_16)
{
    shared_ptr<runtime::he::he_heaan::HEHeaanBackend> he_backend =
        dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(
            runtime::Backend::create("HE_HEAAN"));

    Shape shape_a{10, 100};
    Shape shape_b{100};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});
    Shape shape_r{10};

    // Create some tensors for input/output
    auto a = he_backend->create_tensor(element::f32, shape_a);
    copy_data(
        a,
        vector<float>{
            0,  0,  -1, 1,  -1, 1,  1,  0,  1,  0,  0,  1,  1,  -1, 0,  -1, -1, 1,  1,  0,  1,  0,
            0,  0,  0,  0,  0,  1,  1,  0,  1,  -1, 0,  -1, -1, 0,  0,  0,  1,  -1, 1,  1,  -1, 0,
            -1, 0,  1,  -1, 1,  0,  -1, 1,  0,  0,  0,  -1, 1,  -1, 0,  0,  -1, -1, 0,  0,  -1, 1,
            0,  1,  -1, -1, 0,  -1, 1,  1,  -1, 0,  0,  0,  0,  0,  0,  0,  0,  -1, -1, -1, 0,  0,
            0,  -1, -1, 1,  2,  0,  -1, 0,  0,  0,  0,  1,  1,  0,  0,  1,  0,  0,  0,  1,  0,  0,
            1,  0,  0,  0,  0,  1,  0,  0,  0,  1,  -1, 0,  1,  1,  0,  1,  1,  0,  0,  0,  -1, 0,
            -1, 0,  1,  1,  1,  0,  -1, -1, 0,  1,  0,  0,  1,  0,  0,  1,  -1, -1, 0,  -1, -1, -1,
            0,  0,  -1, 1,  0,  1,  1,  1,  -1, 1,  -1, -1, 0,  0,  0,  2,  -1, 0,  0,  -1, 0,  0,
            -1, 0,  0,  0,  1,  -1, -1, 0,  1,  1,  2,  -1, 0,  -1, 1,  -1, 0,  -1, 0,  0,  -1, 1,
            0,  0,  0,  -1, 0,  1,  0,  -1, 0,  0,  0,  1,  1,  1,  -2, 0,  0,  0,  1,  1,  0,  0,
            2,  0,  -1, 0,  0,  1,  -1, -1, 0,  1,  -1, 0,  -1, 0,  1,  1,  1,  1,  0,  0,  1,  1,
            0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  -1, -1, 0,  1,  1,  0,  -1, 1,  1,  0,  0,
            -1, 0,  1,  1,  0,  -1, 0,  -1, 1,  0,  1,  1,  0,  0,  -2, 0,  1,  1,  -1, 0,  -1, -1,
            0,  0,  0,  -1, -1, 0,  0,  0,  0,  0,  0,  1,  0,  1,  -2, 0,  0,  -1, -1, 0,  1,  0,
            -1, 0,  -1, 0,  -1, 1,  1,  0,  -1, 0,  -1, -1, 1,  0,  0,  -2, 0,  1,  -1, 2,  1,  0,
            -1, 0,  1,  0,  1,  0,  1,  1,  1,  0,  0,  0,  0,  0,  1,  -1, -1, 0,  0,  0,  1,  0,
            0,  0,  0,  1,  1,  0,  0,  2,  0,  -1, 0,  1,  1,  -1, -1, 0,  1,  0,  0,  1,  -1, -1,
            0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  2,  0,  0,  0,  -1, -1, 1,  -1, -1, 1,  0,
            0,  -1, 0,  0,  1,  -1, 1,  1,  0,  -1, 1,  0,  0,  1,  0,  0,  1,  0,  0,  -1, -1, -1,
            0,  0,  -1, -1, 1,  1,  -1, 2,  0,  -1, 0,  0,  1,  1,  -1, 1,  -1, 1,  1,  0,  0,  -1,
            0,  -1, 0,  0,  -1, 1,  -1, 0,  0,  1,  0,  0,  -1, -1, 0,  0,  -3, 0,  0,  -1, 0,  -1,
            -1, -1, 0,  0,  1,  -1, -1, 1,  0,  1,  1,  0,  -1, -1, -1, 0,  0,  0,  1,  0,  0,  1,
            1,  1,  -1, 1,  -1, 1,  1,  0,  -1, 1,  1,  0,  -1, -1, 1,  1,  1,  1,  0,  0,  0,  1,
            0,  0,  -1, 0,  0,  1,  -1, 0,  0,  1,  1,  -1, 0,  1,  -1, 0,  0,  1,  1,  -1, 1,  -1,
            0,  -1, 0,  0,  1,  0,  1,  -2, -2, -1, 0,  0,  1,  -1, 0,  0,  -1, -2, 0,  -1, 0,  0,
            0,  0,  2,  1,  1,  0,  1,  0,  -1, 1,  1,  0,  -1, -1, 1,  1,  1,  -1, 0,  0,  0,  0,
            0,  0,  -1, 0,  0,  -1, 0,  -1, 1,  0,  1,  -1, 0,  2,  1,  -1, 0,  1,  0,  1,  -1, -1,
            1,  0,  2,  0,  -1, 0,  0,  1,  0,  0,  1,  -1, 0,  -1, -1, 0,  1,  1,  1,  1,  -1, 0,
            1,  0,  0,  1,  -1, 0,  1,  0,  0,  0,  -1, 0,  1,  1,  1,  -1, 0,  0,  -1, -2, -1, 1,
            0,  0,  0,  0,  0,  0,  0,  -1, -1, -1, 0,  0,  0,  0,  1,  1,  0,  1,  -1, 0,  0,  -1,
            0,  0,  0,  1,  -1, -2, 1,  -1, 0,  0,  0,  0,  1,  -1, 0,  1,  0,  -1, 0,  0,  0,  1,
            1,  1,  0,  -1, 0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  -1, 0,  1,  -1, -1, -1,
            0,  0,  0,  0,  0,  0,  0,  -1, 1,  0,  -1, -1, 1,  1,  0,  0,  0,  0,  -1, 0,  1,  1,
            0,  0,  0,  -1, -1, 0,  -2, 0,  1,  1,  0,  2,  -1, 0,  0,  0,  0,  0,  0,  1,  0,  2,
            -1, -2, 0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  -1, 0,  1,  -1, 1,  0,  0,  1,  0,  -1,
            -1, 0,  1,  0,  2,  0,  -1, 1,  0,  1,  0,  -2, 1,  0,  -1, 0,  1,  -1, 0,  0,  -1, -1,
            0,  1,  0,  1,  1,  -1, 0,  0,  -1, -1, 0,  1,  -1, -1, -1, 1,  0,  -1, 0,  0,  -2, 0,
            0,  0,  0,  -1, 0,  -1, 1,  1,  1,  1,  -1, -1, 0,  1,  0,  1,  -1, 1,  1,  0,  1,  -1,
            -1, 0,  0,  1,  0,  1,  1,  -1, -1, 1,  0,  0,  1,  -1, 0,  0,  0,  0,  1,  -1, 0,  1,
            0,  -1, 0,  -1, 1,  0,  -1, 1,  1,  -1, 0,  -1, 1,  0,  -2, -1, 1,  0,  0,  0,  0,  0,
            0,  0,  1,  -1, 1,  -2, -1, 0,  1,  0,  1,  1,  1,  -1, -1, 0,  0,  0,  0,  -1, -1, 1,
            0,  1,  0,  1,  0,  0,  0,  0,  -1, 1,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  -1, 0,
            0,  -1, -1, 1,  -1, 0,  0,  0,  1,  0,  -1, 1,  -1, -2, 0,  0,  -1, 1,  1,  0,  1,  1,
            1,  -1, 0,  1,  -1, -1, 0,  0,  -1, 0,  -1, 0,  0,  1,  0,  0,  -1, 0,  0,  1,  0,  1,
            1,  0,  0,  0,  -1, 0,  -1, 1,  0,  1,  1,  -1, -1, -1, 0,  -1, -1, 1,  -1, 0,  0,  1,
            1,  -1, 0,  1,  -1, 0,  -2, 0,  0,  0});

    auto b = he_backend->create_tensor(element::f32, shape_b);
    copy_data(b,
              vector<float>{
                  16,  256,  64,   676, 121,  1681, 81,  441,  9,    49,  36,  324, 256,
                  529, 144,  4,    256, 841,  25,   196, 64,   169,  121, 289, 289, 169,
                  4,   1024, 169,  361, 2209, 1521, 4,   484,  1369, 361, 841, 361, 25,
                  441, 529,  961,  81,  25,   784,  121, 9,    1296, 784, 16,  324, 49,
                  144, 484,  196,  400, 289,  4,    4,   400,  144,  36,  1,   81,  1024,
                  484, 289,  676,  64,  324,  9,    36,  1156, 36,   961, 100, 121, 36,
                  441, 25,   121,  4,   484,  25,   324, 144,  625,  64,  25,  256, -2.51364e-14,
                  49,  484,  1444, 441, 36,   1296, 9,   529,  256});
    auto result = he_backend->create_tensor(element::f32, shape_r);

    he_backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{2173, 944, 1151, 1723, -1674, 569, -1985, 9776, -4997, -1903}),
              read_vector<float>(result));
}

TEST_F(TestHEBackend, tf_mnist_const_1)
{
    shared_ptr<runtime::he::he_heaan::HEHeaanBackend> backend =
        dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(
            runtime::Backend::create("HE_HEAAN"));
    //shared_ptr<runtime::he::he_seal::HESealBackend> backend = dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(runtime::Backend::create("HE_SEAL"));
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

TEST_F(TestHEBackend, tf_mnist_const_1_int)
{
    auto backend = runtime::Backend::create("HE");
    const string json_path =
        file_util::path_join(HE_SERIALIZED_ZOO, "mnist_mlp_const_1_inputs_int.js");
    const string json_string = file_util::read_file_to_string(json_path);
    shared_ptr<Function> f = deserialize(json_string);

    auto parameters = f->get_parameters();
    vector<shared_ptr<runtime::TensorView>> parameter_tvs;
    for (auto parameter : parameters)
    {
        auto& shape = parameter->get_shape();
        auto& type = parameter->get_element_type();
        auto parameter_tv = backend->create_tensor(type, shape);
        NGRAPH_INFO << "created tensor of " << shape_size(shape) << " elements";
        copy_data(parameter_tv, vector<int64_t>(shape_size(shape)));
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

    EXPECT_EQ((vector<int64_t>{2173, 944, 1151, 1723, -1674, 569, -1985, 9776, -4997, -1903}),
              read_vector<int64_t>(result_tvs[0]));
}

TEST_F(TestHEBackend, tf_mnist_const_5)
{
    auto backend = runtime::Backend::create("HE");
    const string json_path = file_util::path_join(HE_SERIALIZED_ZOO, "mnist_mlp_const_5_inputs.js");
    const string json_string = file_util::read_file_to_string(json_path);
    shared_ptr<Function> f = deserialize(json_string);

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

    EXPECT_EQ(
        (vector<float>{2173,  -115,  -4823, 12317,  1581,   944,   4236,   13188, 4436,  2140,
                       1151,  18967, 3295,  659,    -21,    1723,  6923,   -3925, -3237, -637,
                       -1674, -3530, 7586,  -1818,  11578,  569,   -11907, -2731, -91,   -3363,
                       -1985, 2383,  -1781, 8035,   1183,   9776,  -4648,  620,   1244,  768,
                       -4997, 1847,  -8089, -12449, -11285, -1903, -8727,  -1791, -4979, 3849}),
        read_vector<float>(result_tvs[0]));
}

/* TEST_F(TestHEBackend, tf_ptb_const_1)
{
    auto backend = runtime::Backend::create("CPU");
    const string json_path = file_util::path_join(HE_SERIALIZED_ZOO, "ptb_rnn_const_2_2_3.js");
    const string json_string = file_util::read_file_to_string(json_path);
    shared_ptr<Function> f = deserialize(json_string);

    // Visualize model
    auto model_file_name =
        "ptb_rnn_const_2_2_3" + string(".") + pass::VisualizeTree::get_file_ext();

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

    EXPECT_EQ(1, 2); // TODO: add expected results
}

TEST_F(TestHEBackend, tf_mnist_rnn_const)
{
    auto backend = runtime::Backend::create("HE");
    string filename = "mnist_rnn_batch2_layer8_skipeven_binary";
    const string json_path = file_util::path_join(HE_SERIALIZED_ZOO, filename + ".js");
    const string json_string = file_util::read_file_to_string(json_path);
    shared_ptr<Function> f = deserialize(json_string);

    // Visualize model
    auto model_file_name = filename + string(".") + pass::VisualizeTree::get_file_ext();

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
    NGRAPH_INFO << "num results " << result_tvs.size();

    auto result = read_vector<float>(result_tvs[0]); // TODO: clean up
    auto expect = vector<float>{
        -7.88143, -5.8131,  -3.96331, -1.21652, -18.0735,  -8.02312, -30.0687, 13.0319,  -2.70154,
        0.718744, 0.37304,  15.3126,  27.5812,  23.4542,   -2.33102, 7.44211,  -1.77696, 6.01429,
        16.2065,  6.80028,  5.75904,  42.5145,  -0.780159, 2.56209,  12.7461,  22.2679,  12.6967,
        16.0024,  17.6427,  18.5718,  29.0376,  14.2316,   11.0897,  -9.36387, 3.29604,  5.10122,
        8.69652,  1.76425,  2.78079,  -2.29045, 11.1611,   21.6455,  13.9041,  8.51218,  65.6764,
        17.5173,  -22.6419, 25.9811,  32.2385,  41.8058};
    for (int i = 0; i < result.size(); ++i)
    {
        EXPECT_NEAR(expect[i], result[i], 0.0001);
    }
} */
