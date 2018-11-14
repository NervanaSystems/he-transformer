//*****************************************************************************
// Copyright 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <assert.h>

#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

#include "he_backend.hpp"
#include "test_util.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, broadcast_vector)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{4};
    auto t = make_shared<op::Broadcast>(A, shape_r, AxisSet{0});
    auto f = make_shared<Function>(t, op::ParameterVector{A});

    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend, true);

    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{6});

        backend->call(f, {result}, {a});
        EXPECT_TRUE(all_close((vector<float>{6, 6, 6, 6}), read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_to_non_existent_axis)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{4};
    ASSERT_THROW(auto f = make_shared<Function>(
                     make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1}), op::ParameterVector{A}),
                 ngraph_error);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_matrix)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2};
    auto t = make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend, true);

    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{6});

        backend->call(f, {result}, {a});
        EXPECT_TRUE(all_close((vector<float>{6, 6, 6, 6}), read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_tensor)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto t = make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1, 2});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend, true);

    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{6});

        backend->call(f, {result}, {a});
        EXPECT_TRUE(all_close((vector<float>{6, 6, 6, 6, 6, 6, 6, 6}), read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_trivial)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Broadcast>(A, shape, AxisSet{});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend, true);

    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{2, 4, 6, 8, 16, 32, 64, 128});

        backend->call(f, {result}, {a});
        EXPECT_TRUE(
            all_close((vector<float>{2, 4, 6, 8, 16, 32, 64, 128}), read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_vector_colwise)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 4};
    auto t = make_shared<op::Broadcast>(A, shape_r, AxisSet{1});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend, true);

    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{1, 2, 3});

        backend->call(f, {result}, {a});
        EXPECT_TRUE(all_close((vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}),
                              read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_vector_rowwise)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 4};
    auto t = make_shared<op::Broadcast>(A, shape_r, AxisSet{0});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend, true);

    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{1, 2, 3, 4});

        backend->call(f, {result}, {a});
        EXPECT_TRUE(all_close((vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}),
                              read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_matrix_0)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto t = make_shared<op::Broadcast>(A, shape_r, AxisSet{0});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend, true);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{1, 2, 3, 4});

        backend->call(f, {result}, {a});
        EXPECT_TRUE(all_close((vector<float>{1, 2, 3, 4, 1, 2, 3, 4}), read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_matrix_1)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto t = make_shared<op::Broadcast>(A, shape_r, AxisSet{1});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend, true);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{1, 2, 3, 4});

        backend->call(f, {result}, {a});
        EXPECT_TRUE(all_close((vector<float>{1, 2, 1, 2, 3, 4, 3, 4}), read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_matrix_2)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto t = make_shared<op::Broadcast>(A, shape_r, AxisSet{2});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend, true);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{1, 2, 3, 4});

        backend->call(f, {result}, {a});
        EXPECT_TRUE(all_close((vector<float>{1, 1, 2, 2, 3, 3, 4, 4}), read_vector<float>(result)));
    }
}