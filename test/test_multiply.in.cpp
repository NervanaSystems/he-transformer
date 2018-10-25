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

#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

#include "he_backend.hpp"
// #include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/bfv/he_seal_bfv_backend.hpp"

#include "test_util.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

TEST(${BACKEND_NAME}, multiply_2_3)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 3};
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, op::ParameterVector{a, b});

    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend);

    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto t_a = inputs[0];
        auto t_b = inputs[1];
        auto t_result = results[0];

        copy_he_data(t_a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector(), backend);
        copy_he_data(t_b, test::NDArray<float, 2>({{7, 8, 9}, {10, 11, 12}}).get_vector(), backend);
        backend->call(f, {t_result}, {t_a, t_b});
        EXPECT_EQ(read_he_vector<float>(t_result, backend),
                  (test::NDArray<float, 2>({{7, 16, 27}, {40, 55, 72}})).get_vector());
    }
}

TEST(${BACKEND_NAME}, square_2_3)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 3};
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Multiply>(a, a);
    auto f = make_shared<Function>(t, op::ParameterVector{a, a});

    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {a,a }, backend);

    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto t_a = inputs[0];
        auto t_b = inputs[1];
        auto t_result = results[0];

        copy_he_data(t_a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector(), backend);
        backend->call(f, {t_result}, {t_a, t_b});
        EXPECT_EQ(read_he_vector<float>(t_result, backend),
                  (test::NDArray<float, 2>({{1, 4, 9}, {16, 25, 36}})).get_vector());
    }
}

TEST(${BACKEND_NAME}, multiply_optimized_2_3)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 3};
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, op::ParameterVector{a, b});

    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend);

    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto t_a = inputs[0];
        auto t_b = inputs[1];
        auto t_result = results[0];

        copy_he_data(t_a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector(), backend);
        copy_he_data(t_b, test::NDArray<float, 2>({{-1, 0, 1}, {-1, 0, 1 }}).get_vector(), backend);
        backend->call(f, {t_result}, {t_a, t_b});
        EXPECT_EQ(read_he_vector<float>(t_result, backend),
                  (test::NDArray<float, 2>({{-1, 0, 3}, {-4, 0, 6}})).get_vector());
    }
}