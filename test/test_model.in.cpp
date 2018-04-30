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

TEST_F(TestHEBackend, tf_mnist_const_1)
{
    auto backend = runtime::Backend::create("HE");
    const string json_path = file_util::path_join(HE_SERIALIZED_ZOO, "mnist_mlp_const_1_inputs.js");
    const string json_string = file_util::read_file_to_string(json_path);
    shared_ptr<Function> f = deserialize(json_string);

    auto parameters = f->get_parameters();
    vector<shared_ptr<runtime::TensorView>> parameter_tvs;
    for (auto parameter : parameters)
    {
        auto& shape = parameter->get_shape();
        auto& type = parameter->get_element_type();
        auto parameter_tv = backend->create_tensor(type, shape);
        NGRAPH_INFO << "created tensor ";
        NGRAPH_INFO << "elements " << shape_size(shape);
        copy_data(parameter_tv, vector<float>(shape_size(shape)));
        NGRAPH_INFO << "copied " << shape_size(shape);
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

// TODO: Need to tune parameters so this test passes
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
        NGRAPH_INFO << "created tensor ";
        NGRAPH_INFO << "elements " << shape_size(shape);
        copy_data(parameter_tv, vector<float>(shape_size(shape)));
        NGRAPH_INFO << "copied " << shape_size(shape);
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
