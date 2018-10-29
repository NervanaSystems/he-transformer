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
#include "he_heaan_backend.hpp"
#include "he_seal_backend.hpp"

#include "test_util.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, trivial)
{
    int a = 1;
    int b = 2;
    EXPECT_EQ(3, a + b);
}

NGRAPH_TEST(${BACKEND_NAME}, backend_init)
{
    auto he_seal = runtime::Backend::create("HE:SEAL:BFV");
    auto he_heaan = runtime::Backend::create("HE:SEAL:CKKS");
    EXPECT_EQ(1, 1);
}


NGRAPH_TEST(${BACKEND_NAME}, ab_batch)
{
    auto backend = static_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    Shape shape_a{4, 3};
    Shape shape_b{4, 3};
    Shape shape_r{4, 3};
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto b = make_shared<op::Parameter>(element::f32, shape_b);
    auto t = make_shared<op::Add>(a, b);

    auto f = make_shared<Function>(t, op::ParameterVector{a, b});

    // Create some tensors for input/output
    auto t_a = backend->create_tensor(element::f32, shape_a, true);
    auto t_b = backend->create_tensor(element::f32, shape_b, true);
    auto t_result = backend->create_tensor(element::f32, shape_r, true);

    copy_data(t_a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    copy_data(t_b, vector<float>{13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    backend->call(f, {t_result}, {t_a, t_b});
    EXPECT_EQ((vector<float>{14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36}),
              generalized_read_vector<float>(t_result));
}

NGRAPH_TEST(${BACKEND_NAME}, ab_batch_multiply)
{
    auto backend = static_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    Shape shape_a{2, 1};
    Shape shape_b{2, 1};
    Shape shape_r{2, 1};
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto b = make_shared<op::Parameter>(element::f32, shape_b);
    auto t = make_shared<op::Multiply>(a, b);

    auto f = make_shared<Function>(t, op::ParameterVector{a, b});

    // Create some tensors for input/output
    auto t_a = backend->create_tensor(element::f32, shape_a, true);
    auto t_b = backend->create_plain_tensor(element::f32, shape_b);
    auto t_result = backend->create_tensor(element::f32, shape_r, true);

    copy_data(t_a, vector<float>{1, 2});
    copy_data(t_b, vector<float>{3, 3});
    backend->call(f, {t_result}, {t_a, t_b});
    EXPECT_EQ((vector<float>{3, 6}), generalized_read_vector<float>(t_result));
}

NGRAPH_TEST(${BACKEND_NAME}, ab_batch_dot)
{
    auto backend = static_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    Shape shape_a{2, 1};
    Shape shape_b{1};
    Shape shape_r{2, 1};
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto b = make_shared<op::Parameter>(element::f32, shape_b);
    auto t = make_shared<op::Dot>(a, b);

    auto f = make_shared<Function>(t, op::ParameterVector{a, b});

    // Create some tensors for input/output
    auto t_a = backend->create_tensor(element::f32, shape_a, true);
    auto t_b = backend->create_plain_tensor(element::f32, shape_b);
    auto t_result = backend->create_tensor(element::f32, shape_r, true);

    copy_data(t_a, vector<float>{1, 2});
    copy_data(t_b, vector<float>{3});
    backend->call(f, {t_result}, {t_a, t_b});
    EXPECT_EQ((vector<float>{3, 6}), generalized_read_vector<float>(t_result));
}

NGRAPH_TEST(${BACKEND_NAME}, ab)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 3};
    auto a = make_shared<op::Parameter>(element::i64, shape);
    auto b = make_shared<op::Parameter>(element::i64, shape);
    auto t = make_shared<op::Add>(a, b);
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

        copy_data(t_a, test::NDArray<int64_t, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
        copy_data(t_b, test::NDArray<int64_t, 2>({{7, 8, 9}, {10, 11, 12}}).get_vector());
        backend->call(f, {t_result}, {t_a, t_b});
        EXPECT_EQ(read_vector<int64_t>(t_result),
                  (test::NDArray<int64_t, 2>({{8, 10, 12}, {14, 16, 18}})).get_vector());
    }
}

NGRAPH_TEST(${BACKEND_NAME}, ab_square)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 3};
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Multiply>(a, a);
    auto f = make_shared<Function>(t, op::ParameterVector{a, a});

    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {a, a}, backend);

    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto t_a = inputs[0];
        auto t_result = results[0];

        copy_data(t_a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
        backend->call(f, {t_result}, {t_a, t_a});
        EXPECT_EQ(read_vector<float>(t_result),
                  (test::NDArray<float, 2>({{1, 4, 9}, {16, 25, 36}})).get_vector());
    }
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_optimized)
{
    auto backend = static_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    Shape shape{6};
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::Parameter>(element::f32, shape);
    auto c = make_shared<op::Parameter>(element::f32, shape);
    auto t = (a * b) + c;

    auto f = make_shared<Function>(t, op::ParameterVector{a, b, c});

    // Create some tensors for input/output
    auto t_a = backend->create_tensor(element::f32, shape);
    auto t_b = backend->create_plain_tensor(element::f32, shape);
    auto t_c = backend->create_tensor(element::f32, shape);
    auto t_result = backend->create_tensor(element::f32, shape);

    copy_data(t_a, vector<float>{1, -2, 3, -4, 5, -6});
    copy_data(t_b, vector<float>{-1, -1, 0, 0, 1, 1});
    copy_data(t_c, vector<float>{0, 0, 1, 0, 0, 0});
    backend->call(f, {t_result}, {t_a, t_b, t_c});
    EXPECT_TRUE(test::all_close(vector<float>{-1, 2, 1, 0, 5, -6}, read_vector<float>(t_result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_optimized_small)
{
    auto backend = static_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    Shape shape_a{1, 3};
    Shape shape_b{3, 1};
    Shape shape_r{1, 1};
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto b = make_shared<op::Parameter>(element::f32, shape_b);
    auto r = make_shared<op::Parameter>(element::f32, shape_r);
    auto t = make_shared<op::Dot>(a, b);
    auto f = make_shared<Function>(t, op::ParameterVector{a, b});

    // Create some tensors for input/output
    auto t_a = backend->create_tensor(element::f32, shape_a);
    auto t_b = backend->create_plain_tensor(element::f32, shape_b);
    auto t_result = backend->create_tensor(element::f32, shape_r);

    copy_data(t_a, vector<float>{1, 2, 3});
    copy_data(t_b, vector<float>{0, 1, 0});
    backend->call(f, {t_result}, {t_a, t_b});
    EXPECT_TRUE(test::all_close(vector<float>{2}, read_vector<float>(t_result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_optimized)
{
    auto backend = static_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    Shape shape_a{2, 3};
    Shape shape_b{3, 4};
    Shape shape_r{2, 4};
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto b = make_shared<op::Parameter>(element::f32, shape_b);
    auto r = make_shared<op::Parameter>(element::f32, shape_r);
    auto t = make_shared<op::Dot>(a, b);

    auto f = make_shared<Function>(t, op::ParameterVector{a, b});

    // Create some tensors for input/output
    auto t_a = backend->create_tensor(element::f32, shape_a);
    auto t_b = backend->create_plain_tensor(element::f32, shape_b);
    auto t_result = backend->create_tensor(element::f32, shape_r);

    copy_data(t_a, vector<float>{1, -2, 3, -4, 5, -6});
    copy_data(t_b, vector<float>{-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1});
    backend->call(f, {t_result}, {t_a, t_b});
    EXPECT_EQ((vector<float>{2, -5, 3, 2, -2, 11, -9, -2}), read_vector<float>(t_result));
}

NGRAPH_TEST(${BACKEND_NAME}, add_zero)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Add>(A, B);
    auto f = make_shared<Function>(t, op::ParameterVector{A, B});

    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {A, B}, backend);

    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto b = inputs[1];
        auto result = results[0];

        copy_data(a, vector<float>{1, 2, 3, 4});
        copy_data(b, vector<float>{0, 0, 0, 0});
        backend->call(f, {result}, {a, b});
        EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, subtract)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Subtract>(A, B);
    auto f = make_shared<Function>(t, op::ParameterVector{A, B});

    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {A, B}, backend);

    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto b = inputs[1];
        auto result = results[0];

        copy_data(a, vector<float>{8, 6, 4, 2});
        copy_data(b, vector<float>{1, 2, 3, 4});
        backend->call(f, {result}, {a, b});
        EXPECT_EQ((vector<float>{7, 4, 1, -2}), read_vector<float>(result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, subtract_from_zero)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Subtract>(A, B);
    auto f = make_shared<Function>(t, op::ParameterVector{A, B});

    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {A, B}, backend);

    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto b = inputs[1];
        auto result = results[0];

        copy_data(a, vector<float>{0, 0, 0, 0});
        copy_data(b, vector<float>{1, 2, 3, 4});
        backend->call(f, {result}, {a, b});
        EXPECT_EQ((vector<float>{-1, -2, -3, -4}), read_vector<float>(result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, subtract_zero)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Subtract>(A, B);
    auto f = make_shared<Function>(t, op::ParameterVector{A, B});

    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {A, B}, backend);

    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto b = inputs[1];
        auto result = results[0];

        copy_data(a, vector<float>{1, 2, 3, 4});
        copy_data(b, vector<float>{0, 0, 0, 0});
        backend->call(f, {result}, {a, b});
        EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, abc)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto t = (A + B) * C;
    auto f = make_shared<Function>(t, op::ParameterVector{A, B, C});

    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {A, B, C}, backend);

    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto b = inputs[1];
        auto c = inputs[2];
        auto result = results[0];

        copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
        copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
        copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());
        backend->call(f, {result}, {a, b, c});

        EXPECT_EQ(read_vector<float>(result),
                  (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

        backend->call(f, {result}, {b, a, c});
        EXPECT_EQ(read_vector<float>(result),
                  (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

        backend->call(f, {result}, {a, c, b});
        EXPECT_EQ(read_vector<float>(result),
                  (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());
    }
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 3};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 3};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 8};
    auto r = make_shared<op::Concat>(NodeVector{A, B, C}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B, C});
    // Create some tensors for input/output
    bool consistent_type = true;
    auto tensors_list = generate_plain_cipher_tensors({r}, {A, B, C}, backend, consistent_type);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto b = inputs[1];
        auto c = inputs[2];
        auto result = results[0];

        copy_data(a, vector<float>{2, 4, 8, 16});
        copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
        copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});

        backend->call(f, {result}, {a, b, c});
        EXPECT_EQ((vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}),
                  read_vector<float>(result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowwise)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{3, 2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{8, 2};
    auto r = make_shared<op::Concat>(NodeVector{A, B, C}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B, C});

    bool consistent_type = true;
    auto tensors_list = generate_plain_cipher_tensors({r}, {A, B, C}, backend, consistent_type);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto b = inputs[1];
        auto c = inputs[2];
        auto result = results[0];

        copy_data(a, vector<float>{2, 4, 8, 16});
        copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
        copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});

        backend->call(f, {result}, {a, b, c});
        EXPECT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
                  read_vector<float>(result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_int64)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::i64, shape_b);
    Shape shape_c{3, 2};
    auto C = make_shared<op::Parameter>(element::i64, shape_c);
    Shape shape_r{8, 2};
    auto r = make_shared<op::Concat>(NodeVector{A, B, C}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B, C});

    bool consistent_type = true;
    auto tensors_list = generate_plain_cipher_tensors({r}, {A, B, C}, backend, consistent_type);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto b = inputs[1];
        auto c = inputs[2];
        auto result = results[0];

        copy_data(a, vector<int64_t>{2, 4, 8, 16});
        copy_data(b, vector<int64_t>{1, 2, 4, 8, 16, 32});
        copy_data(c, vector<int64_t>{2, 3, 5, 7, 11, 13});

        backend->call(f, {result}, {a, b, c});
        EXPECT_EQ((vector<int64_t>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
                  read_vector<int64_t>(result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{6};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{12};
    auto r = make_shared<op::Concat>(NodeVector{A, B, C}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B, C});

    bool consistent_type = true;
    auto tensors_list = generate_plain_cipher_tensors({r}, {A, B, C}, backend, consistent_type);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto b = inputs[1];
        auto c = inputs[2];
        auto result = results[0];

        copy_data(a, vector<float>{2, 4, 8, 16});
        copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
        copy_data(c, vector<float>{18, 19});

        backend->call(f, {result}, {a, b, c});
        EXPECT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}),
                  read_vector<float>(result));
    }
}

// from numpy import *
// a=linspace(1,2*3*4*3*2,2*3*4*3*2)
// b=linspace(1000+1,1000+2*3*3*3*2,2*3*3*3*2)
// c=linspace(2000+1,2000+2*3*2*3*2,2*3*2*3*2)
// a.shape=(2,3,4,3,2)
// b.shape=(2,3,3,3,2)
// c.shape=(2,3,2,3,2)
// z=concatenate((a,b,c),axis=2)
// z.shape=(2*3*(4+3+2)*3*2)
// set_printoptions(suppress=True)
// print(z)
//
// [    1.     2.     3.     4.     5.     6.     7.     8.     9.    10.
//     11.    12.    13.    14.    15.    16.    17.    18.    19.    20.
//     21.    22.    23.    24.  1001.  1002.  1003.  1004.  1005.  1006.
//   1007.  1008.  1009.  1010.  1011.  1012.  1013.  1014.  1015.  1016.
//   1017.  1018.  2001.  2002.  2003.  2004.  2005.  2006.  2007.  2008.
//   2009.  2010.  2011.  2012.    25.    26.    27.    28.    29.    30.
//     31.    32.    33.    34.    35.    36.    37.    38.    39.    40.
//     41.    42.    43.    44.    45.    46.    47.    48.  1019.  1020.
//   1021.  1022.  1023.  1024.  1025.  1026.  1027.  1028.  1029.  1030.
//   1031.  1032.  1033.  1034.  1035.  1036.  2013.  2014.  2015.  2016.
//   2017.  2018.  2019.  2020.  2021.  2022.  2023.  2024.    49.    50.
//     51.    52.    53.    54.    55.    56.    57.    58.    59.    60.
//     61.    62.    63.    64.    65.    66.    67.    68.    69.    70.
//     71.    72.  1037.  1038.  1039.  1040.  1041.  1042.  1043.  1044.
//   1045.  1046.  1047.  1048.  1049.  1050.  1051.  1052.  1053.  1054.
//   2025.  2026.  2027.  2028.  2029.  2030.  2031.  2032.  2033.  2034.
//   2035.  2036.    73.    74.    75.    76.    77.    78.    79.    80.
//     81.    82.    83.    84.    85.    86.    87.    88.    89.    90.
//     91.    92.    93.    94.    95.    96.  1055.  1056.  1057.  1058.
//   1059.  1060.  1061.  1062.  1063.  1064.  1065.  1066.  1067.  1068.
//   1069.  1070.  1071.  1072.  2037.  2038.  2039.  2040.  2041.  2042.
//   2043.  2044.  2045.  2046.  2047.  2048.    97.    98.    99.   100.
//    101.   102.   103.   104.   105.   106.   107.   108.   109.   110.
//    111.   112.   113.   114.   115.   116.   117.   118.   119.   120.
//   1073.  1074.  1075.  1076.  1077.  1078.  1079.  1080.  1081.  1082.
//   1083.  1084.  1085.  1086.  1087.  1088.  1089.  1090.  2049.  2050.
//   2051.  2052.  2053.  2054.  2055.  2056.  2057.  2058.  2059.  2060.
//    121.   122.   123.   124.   125.   126.   127.   128.   129.   130.
//    131.   132.   133.   134.   135.   136.   137.   138.   139.   140.
//    141.   142.   143.   144.  1091.  1092.  1093.  1094.  1095.  1096.
//   1097.  1098.  1099.  1100.  1101.  1102.  1103.  1104.  1105.  1106.
//   1107.  1108.  2061.  2062.  2063.  2064.  2065.  2066.  2067.  2068.
//   2069.  2070.  2071.  2072.]
NGRAPH_TEST(${BACKEND_NAME}, concat_5d)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    vector<float> a_data(2 * 3 * 4 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 4 * 3 * 2; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(2 * 3 * 3 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 3 * 3 * 2; i++)
    {
        b_data[i] = 1000 + float(i + 1);
    }

    vector<float> c_data(2 * 3 * 2 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 2 * 3 * 2; i++)
    {
        c_data[i] = 2000 + float(i + 1);
    }

    Shape shape_a{2, 3, 4, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 3, 3, 3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 3, 2, 3, 2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 3, 9, 3, 2};

    auto r = make_shared<op::Concat>(NodeVector{A, B, C}, 2);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B, C});
    bool consistent_type = true;
    auto tensors_list = generate_plain_cipher_tensors({r}, {A, B, C}, backend, consistent_type);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto b = inputs[1];
        auto c = inputs[2];
        auto result = results[0];

        copy_data(a, a_data);
        copy_data(b, b_data);
        copy_data(c, c_data);

        backend->call(f, {result}, {a, b, c});
        EXPECT_EQ((vector<float>{
                      1.,    2.,    3.,    4.,    5.,    6.,    7.,    8.,    9.,    10.,   11.,
                      12.,   13.,   14.,   15.,   16.,   17.,   18.,   19.,   20.,   21.,   22.,
                      23.,   24.,   1001., 1002., 1003., 1004., 1005., 1006., 1007., 1008., 1009.,
                      1010., 1011., 1012., 1013., 1014., 1015., 1016., 1017., 1018., 2001., 2002.,
                      2003., 2004., 2005., 2006., 2007., 2008., 2009., 2010., 2011., 2012., 25.,
                      26.,   27.,   28.,   29.,   30.,   31.,   32.,   33.,   34.,   35.,   36.,
                      37.,   38.,   39.,   40.,   41.,   42.,   43.,   44.,   45.,   46.,   47.,
                      48.,   1019., 1020., 1021., 1022., 1023., 1024., 1025., 1026., 1027., 1028.,
                      1029., 1030., 1031., 1032., 1033., 1034., 1035., 1036., 2013., 2014., 2015.,
                      2016., 2017., 2018., 2019., 2020., 2021., 2022., 2023., 2024., 49.,   50.,
                      51.,   52.,   53.,   54.,   55.,   56.,   57.,   58.,   59.,   60.,   61.,
                      62.,   63.,   64.,   65.,   66.,   67.,   68.,   69.,   70.,   71.,   72.,
                      1037., 1038., 1039., 1040., 1041., 1042., 1043., 1044., 1045., 1046., 1047.,
                      1048., 1049., 1050., 1051., 1052., 1053., 1054., 2025., 2026., 2027., 2028.,
                      2029., 2030., 2031., 2032., 2033., 2034., 2035., 2036., 73.,   74.,   75.,
                      76.,   77.,   78.,   79.,   80.,   81.,   82.,   83.,   84.,   85.,   86.,
                      87.,   88.,   89.,   90.,   91.,   92.,   93.,   94.,   95.,   96.,   1055.,
                      1056., 1057., 1058., 1059., 1060., 1061., 1062., 1063., 1064., 1065., 1066.,
                      1067., 1068., 1069., 1070., 1071., 1072., 2037., 2038., 2039., 2040., 2041.,
                      2042., 2043., 2044., 2045., 2046., 2047., 2048., 97.,   98.,   99.,   100.,
                      101.,  102.,  103.,  104.,  105.,  106.,  107.,  108.,  109.,  110.,  111.,
                      112.,  113.,  114.,  115.,  116.,  117.,  118.,  119.,  120.,  1073., 1074.,
                      1075., 1076., 1077., 1078., 1079., 1080., 1081., 1082., 1083., 1084., 1085.,
                      1086., 1087., 1088., 1089., 1090., 2049., 2050., 2051., 2052., 2053., 2054.,
                      2055., 2056., 2057., 2058., 2059., 2060., 121.,  122.,  123.,  124.,  125.,
                      126.,  127.,  128.,  129.,  130.,  131.,  132.,  133.,  134.,  135.,  136.,
                      137.,  138.,  139.,  140.,  141.,  142.,  143.,  144.,  1091., 1092., 1093.,
                      1094., 1095., 1096., 1097., 1098., 1099., 1100., 1101., 1102., 1103., 1104.,
                      1105., 1106., 1107., 1108., 2061., 2062., 2063., 2064., 2065., 2066., 2067.,
                      2068., 2069., 2070., 2071., 2072.}),
                  read_vector<float>(result));
    }
}

// Trivial case with no summed axes.
NGRAPH_TEST(${BACKEND_NAME}, sum_trivial)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Sum>(A, AxisSet{});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{1, 2, 3, 4});

        backend->call(f, {result}, {a});
        EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
    }
}

// Failure has been reported at 5D for some reason
NGRAPH_TEST(${BACKEND_NAME}, sum_trivial_5d)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Sum>(A, AxisSet{});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

        backend->call(f, {result}, {a});
        EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
                  read_vector<float>(result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_to_scalar)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Sum>(A, AxisSet{0, 1});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{1, 2, 3, 4});

        backend->call(f, {result}, {a});
        EXPECT_EQ((vector<float>{10}), read_vector<float>(result));
        // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
        // input tensors, so let's do this too.
        EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_columns)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto t = make_shared<op::Sum>(A, AxisSet{0});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});

        backend->call(f, {result}, {a});
        EXPECT_EQ((vector<float>{9, 12}), read_vector<float>(result));

        // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
        // input tensors, so let's do this too.
        EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_rows)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto t = make_shared<op::Sum>(A, AxisSet{1});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});

        backend->call(f, {result}, {a});
        EXPECT_EQ((vector<float>{3, 7, 11}), read_vector<float>(result));

        // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
        // input tensors, so let's do this too.
        EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_rows_zero)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto t = make_shared<op::Sum>(A, AxisSet{1});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{});

        backend->call(f, {result}, {a});
        EXPECT_TRUE(test::all_close((vector<float>{0, 0, 0}), read_vector<float>(result)));

        // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
        // input tensors, so let's do this too.
        EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_cols_zero)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto t = make_shared<op::Sum>(A, AxisSet{0});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(result, vector<float>({3, 3}));
        backend->call(f, {result}, {a});
        EXPECT_TRUE(test::all_close((vector<float>{0, 0}), read_vector<float>(result)));

        // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
        // input tensors, so let's do this too.
        EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_vector_zero)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto t = make_shared<op::Sum>(A, AxisSet{0});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(result, vector<float>({3}));

        backend->call(f, {result}, {a});
        EXPECT_TRUE(test::all_close((vector<float>{0}), read_vector<float>(result)));

        // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
        // input tensors, so let's do this too.
        EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_to_zero_by_zero)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto t = make_shared<op::Sum>(A, AxisSet{0, 1});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(result, vector<float>({3}));

        backend->call(f, {result}, {a});
        EXPECT_TRUE(test::all_close((vector<float>{0}), read_vector<float>(result)));

        // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
        // input tensors, so let's do this too.
        EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_to_matrix_most_sig)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto t = make_shared<op::Sum>(A, AxisSet{0});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                   15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});

        backend->call(f, {result}, {a});
        EXPECT_EQ((vector<float>{1 + 10 + 19,
                                 2 + 11 + 20,
                                 3 + 12 + 21,
                                 4 + 13 + 22,
                                 5 + 14 + 23,
                                 6 + 15 + 24,
                                 7 + 16 + 25,
                                 8 + 17 + 26,
                                 9 + 18 + 27}),
                  read_vector<float>(result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_to_matrix_least_sig)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto t = make_shared<op::Sum>(A, AxisSet{2});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                   15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});

        backend->call(f, {result}, {a});
        EXPECT_EQ((vector<float>{1 + 2 + 3,
                                 4 + 5 + 6,
                                 7 + 8 + 9,
                                 10 + 11 + 12,
                                 13 + 14 + 15,
                                 16 + 17 + 18,
                                 19 + 20 + 21,
                                 22 + 23 + 24,
                                 25 + 26 + 27}),
                  read_vector<float>(result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_to_vector)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto t = make_shared<op::Sum>(A, AxisSet{0, 1});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                   15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});

        backend->call(f, {result}, {a});
        EXPECT_EQ((vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25,
                                 2 + 11 + 20 + 5 + 14 + 23 + 8 + 17 + 26,
                                 3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
                  read_vector<float>(result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_to_scalar)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto t = make_shared<op::Sum>(A, AxisSet{0, 1, 2});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                   15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});

        backend->call(f, {result}, {a});
        EXPECT_EQ((vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25 + 2 + 11 + 20 + 5 + 14 +
                                 23 + 8 + 17 + 26 + 3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
                  read_vector<float>(result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_eliminate_zero_dim)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto t = make_shared<op::Sum>(A, AxisSet{1});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];
        // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
        copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

        copy_data(a, vector<float>{});
        backend->call(f, {result}, {a});
        EXPECT_TRUE(test::all_close((vector<float>{0, 0, 0, 0, 0, 0}), read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_5d_to_scalar)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{3, 3, 3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto t = make_shared<op::Sum>(A, AxisSet{0, 1, 2, 3, 4});
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, std::vector<float>(std::pow(3, 5), 1));

        backend->call(f, {result}, {a});
        EXPECT_EQ(std::vector<float>{243.}, read_vector<float>(result));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, create_valued_plaintext)
{
    auto backend = static_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    // Fractional
    {
        float val = 3.14;
        element::Type type = element::f32;
        shared_ptr<runtime::he::HEPlaintext> plaintext =
            backend->create_valued_plaintext(val, type);
        float val_decoded;
        backend->decode(&val_decoded, plaintext, type);
        EXPECT_EQ(val_decoded, val);
    }

    // Integer
    {
        int64_t val = 1;
        element::Type type = element::i64;
        shared_ptr<runtime::he::HEPlaintext> plaintext =
            backend->create_valued_plaintext((float)val, type);
        int64_t val_decoded;
        backend->decode(&val_decoded, plaintext, type);
        EXPECT_EQ(val_decoded, val);
    }
    {
        int64_t val = 0;
        element::Type type = element::i64;
        shared_ptr<runtime::he::HEPlaintext> plaintext =
            backend->create_valued_plaintext((float)val, type);
        int64_t val_decoded;
        backend->decode(&val_decoded, plaintext, type);
        EXPECT_EQ(val_decoded, val);
    }
    {
        int64_t val = -2;
        element::Type type = element::i64;
        shared_ptr<runtime::he::HEPlaintext> plaintext =
            backend->create_valued_plaintext((float)val, type);
        int64_t val_decoded;
        backend->decode(&val_decoded, plaintext, type);
        EXPECT_EQ(val_decoded, val);
    }
}

struct ConvolutionTestData
{
    size_t n{0};
    size_t c{0};
    size_t filter{0};
    size_t kernel_size{0};
    size_t w{0};
    size_t h{0};
    shared_ptr<runtime::TensorView> data_val;
    shared_ptr<runtime::TensorView> data_plain_val;
    shared_ptr<runtime::TensorView> weights_val;
    shared_ptr<runtime::TensorView> weights_plain_val;
    shared_ptr<runtime::TensorView> result_val;
    shared_ptr<runtime::TensorView> result_plain_val;
    vector<float> expected_result_val;

    Shape data_shape;
    Shape weights_shape;
    Shape result_shape;
    shared_ptr<op::Parameter> data;
    shared_ptr<op::Parameter> weights;

    void n1c1h3w3(shared_ptr<runtime::he::he_heaan::HEHeaanBackend> backend)
    {
        n = 1;
        c = 1;
        filter = 1;
        kernel_size = 3;
        w = 3;
        h = w;

        data_shape = Shape{n, c, h, w};
        data = make_shared<op::Parameter>(element::f32, data_shape);

        weights_shape = Shape{filter, c, kernel_size, kernel_size};
        weights = make_shared<op::Parameter>(element::f32, weights_shape);
        result_shape = Shape{n, filter, 1, 1};

        data_plain_val = backend->create_plain_tensor(element::f32, data_shape);
        data_val = backend->create_tensor(element::f32, data_shape);
        copy_data(data_val,
                  vector<float>{-0.67765152f,
                                0.10073948f,
                                0.57595438f,
                                -0.3469252f,
                                -0.22134334f,
                                -1.80471897f,
                                -0.80642909f,
                                1.22033095f,
                                2.23235631f});
        copy_data(data_plain_val,
                  vector<float>{-0.67765152f,
                                0.10073948f,
                                0.57595438f,
                                -0.3469252f,
                                -0.22134334f,
                                -1.80471897f,
                                -0.80642909f,
                                1.22033095f,
                                2.23235631f});
        weights_plain_val = backend->create_plain_tensor(element::f32, weights_shape);
        weights_val = backend->create_tensor(element::f32, weights_shape);
        copy_data(weights_plain_val,
                  vector<float>{0.20070229f,
                                -0.54968649f,
                                -0.19819015f,
                                -0.38577855f,
                                1.37109005f,
                                -0.23789984f,
                                0.14867957f,
                                -0.49851316f,
                                -0.84815776f});
        copy_data(weights_val,
                  vector<float>{0.20070229f,
                                -0.54968649f,
                                -0.19819015f,
                                -0.38577855f,
                                1.37109005f,
                                -0.23789984f,
                                0.14867957f,
                                -0.49851316f,
                                -0.84815776f});

        result_val = backend->create_tensor(element::f32, result_shape);
        result_plain_val = backend->create_plain_tensor(element::f32, result_shape);
        copy_data(result_val, vector<float>{0});
        copy_data(result_plain_val, vector<float>{0});

        expected_result_val = vector<float>{-2.66747372f};
    }
};

NGRAPH_TEST(${BACKEND_NAME}, conv_fprop_n1c1h3w3)
{
    auto backend = static_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    ConvolutionTestData conv_test;
    conv_test.n1c1h3w3(backend);

    auto convolution = make_shared<op::Convolution>(conv_test.data, conv_test.weights);

    auto f =
        make_shared<Function>(convolution, op::ParameterVector{conv_test.data, conv_test.weights});

    // Plain plain -> plain
    backend->call(
        f, {conv_test.result_plain_val}, {conv_test.data_plain_val, conv_test.weights_plain_val});
    auto result_vec = read_vector<float>(conv_test.result_plain_val);
    EXPECT_TRUE(test::all_close(conv_test.expected_result_val,
                                read_vector<float>(conv_test.result_plain_val)));

    // Plain cipher -> cipher
    backend->call(f, {conv_test.result_val}, {conv_test.data_plain_val, conv_test.weights_val});
    result_vec = read_vector<float>(conv_test.result_val);
    EXPECT_TRUE(
        test::all_close(conv_test.expected_result_val, read_vector<float>(conv_test.result_val)));

    // Cipher plain -> cipher
    backend->call(f, {conv_test.result_val}, {conv_test.data_val, conv_test.weights_plain_val});
    result_vec = read_vector<float>(conv_test.result_val);
    EXPECT_TRUE(
        test::all_close(conv_test.expected_result_val, read_vector<float>(conv_test.result_val)));

    // Cipher cipher -> cipher
    backend->call(f, {conv_test.result_val}, {conv_test.data_val, conv_test.weights_val});
    result_vec = read_vector<float>(conv_test.result_val);
    EXPECT_TRUE(
        test::all_close(conv_test.expected_result_val, read_vector<float>(conv_test.result_val)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_1channel_1image_padded)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape_a{1, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 4, 4};
    auto t = make_shared<op::AvgPool>(
        A, window_shape, window_movement_strides, padding_below, padding_above, false);
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, test::NDArray<float, 4>({{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}}}).get_vector());

        backend->call(f, {result}, {a});

        EXPECT_TRUE(
            test::all_close(test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 2, 0.0f / 1},
                                                       {0.0f / 2, 4.0f / 4, 6.0f / 4, 2.0f / 2},
                                                       {2.0f / 2, 5.0f / 4, 5.0f / 4, 2.0f / 2},
                                                       {2.0f / 1, 2.0f / 2, 0.0f / 2, 0.0f / 1}}}})
                                .get_vector(),
                            read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 4, 4};
    auto t = make_shared<op::AvgPool>(
        A, window_shape, window_movement_strides, padding_below, padding_above, false);
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a,
                  test::NDArray<float, 4>(
                      {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                      .get_vector());

        backend->call(f, {result}, {a});

        EXPECT_TRUE(
            test::all_close(test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 2, 0.0f / 1},
                                                       {0.0f / 2, 4.0f / 4, 6.0f / 4, 2.0f / 2},
                                                       {2.0f / 2, 5.0f / 4, 5.0f / 4, 2.0f / 2},
                                                       {2.0f / 1, 2.0f / 2, 0.0f / 2, 0.0f / 1}},
                                                      {{3.0f / 1, 8.0f / 2, 7.0f / 2, 2.0f / 1},
                                                       {5.0f / 2, 10.0f / 4, 16.0f / 4, 11.0f / 2},
                                                       {5.0f / 2, 11.0f / 4, 20.0f / 4, 14.0f / 2},
                                                       {3.0f / 1, 9.0f / 2, 11.0f / 2, 5.0f / 1}}}})
                                .get_vector(),
                            read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_only_below)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto t = make_shared<op::AvgPool>(
        A, window_shape, window_movement_strides, padding_below, padding_above, false);
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a,
                  test::NDArray<float, 4>(
                      {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                      .get_vector());

        backend->call(f, {result}, {a});

        EXPECT_TRUE(test::all_close(test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 2},
                                                               {0.0f / 2, 4.0f / 4, 6.0f / 4},
                                                               {2.0f / 2, 5.0f / 4, 5.0f / 4}},
                                                              {{3.0f / 1, 8.0f / 2, 7.0f / 2},
                                                               {5.0f / 2, 10.0f / 4, 16.0f / 4},
                                                               {5.0f / 2, 11.0f / 4, 20.0f / 4}}}})
                                        .get_vector(),
                                    read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_only_above)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto t = make_shared<op::AvgPool>(
        A, window_shape, window_movement_strides, padding_below, padding_above, false);
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a,
                  test::NDArray<float, 4>(
                      {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                      .get_vector());

        backend->call(f, {result}, {a});

        EXPECT_TRUE(test::all_close(test::NDArray<float, 4>({{{{4.0f / 4, 6.0f / 4, 2.0f / 2},
                                                               {5.0f / 4, 5.0f / 4, 2.0f / 2},
                                                               {2.0f / 2, 0.0f / 2, 0.0f / 1}},
                                                              {{10.0f / 4, 16.0f / 4, 11.0f / 2},
                                                               {11.0f / 4, 20.0f / 4, 14.0f / 2},
                                                               {9.0f / 2, 11.0f / 2, 5.0f / 1}}}})
                                        .get_vector(),
                                    read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_3x3)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 5, 5};
    auto t = make_shared<op::AvgPool>(
        A, window_shape, window_movement_strides, padding_below, padding_above, false);
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a,
                  test::NDArray<float, 4>(
                      {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                      .get_vector());

        backend->call(f, {result}, {a});

        EXPECT_TRUE(test::all_close(
            test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 3, 1.0f / 2, 0.0f / 1},
                                       {0.0f / 2, 4.0f / 4, 6.0f / 6, 6.0f / 4, 2.0f / 2},
                                       {2.0f / 3, 6.0f / 6, 8.0f / 9, 6.0f / 6, 2.0f / 3},
                                       {2.0f / 2, 5.0f / 4, 7.0f / 6, 5.0f / 4, 2.0f / 2},
                                       {2.0f / 1, 2.0f / 2, 2.0f / 3, 0.0f / 2, 0.0f / 1}},
                                      {{3.0f / 1, 8.0f / 2, 10.0f / 3, 7.0f / 2, 2.0f / 1},
                                       {5.0f / 2, 10.0f / 4, 21.0f / 6, 16.0f / 4, 11.0f / 2},
                                       {8.0f / 3, 19.0f / 6, 35.0f / 9, 27.0f / 6, 16.0f / 3},
                                       {5.0f / 2, 11.0f / 4, 25.0f / 6, 20.0f / 4, 14.0f / 2},
                                       {3.0f / 1, 9.0f / 2, 14.0f / 3, 11.0f / 2, 5.0f / 1}}}})
                .get_vector(),
            read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_3x3_strided)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{2, 2};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto t = make_shared<op::AvgPool>(
        A, window_shape, window_movement_strides, padding_below, padding_above, false);
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a,
                  test::NDArray<float, 4>(
                      {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                      .get_vector());

        backend->call(f, {result}, {a});

        EXPECT_TRUE(test::all_close(test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 3, 0.0f / 1},
                                                               {2.0f / 3, 8.0f / 9, 2.0f / 3},
                                                               {2.0f / 1, 2.0f / 3, 0.0f / 1}},
                                                              {{3.0f / 1, 10.0f / 3, 2.0f / 1},
                                                               {8.0f / 3, 35.0f / 9, 16.0f / 3},
                                                               {3.0f / 1, 14.0f / 3, 5.0f / 1}}}})
                                        .get_vector(),
                                    read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_3x3_strided_uneven)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{2, 3};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 2};
    auto t = make_shared<op::AvgPool>(
        A, window_shape, window_movement_strides, padding_below, padding_above, false);
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a,
                  test::NDArray<float, 4>(
                      {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                      .get_vector());

        backend->call(f, {result}, {a});

        EXPECT_TRUE(test::all_close(
            test::NDArray<float, 4>(
                {{{{0.0f / 1, 1.0f / 2}, {2.0f / 3, 6.0f / 6}, {2.0f / 1, 0.0f / 2}},
                  {{3.0f / 1, 7.0f / 2}, {8.0f / 3, 27.0f / 6}, {3.0f / 1, 11.0f / 2}}}})
                .get_vector(),
            read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, negative)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Negative>(A);
    auto f = make_shared<Function>(t, op::ParameterVector{A});
    auto tensors_list = generate_plain_cipher_tensors({t}, {A}, backend);
    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto a = inputs[0];
        auto result = results[0];

        copy_data(a, vector<float>{1, -2, 0, -4.75f, 8.75f, -8.75f});

        backend->call(f, {result}, {a});

        EXPECT_TRUE(test::all_close(vector<float>{-1, 2, 0, 4.75f, -8.75f, 8.75f},
                                    read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_count)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call_with_validate(f, {c}, {a}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_type)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::i32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call_with_validate(f, {c}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_shape)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, {2, 3});
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call_with_validate(f, {c}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_count)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);
    auto d = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call_with_validate(f, {c, d}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_type)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::i32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call_with_validate(f, {a}, {b, c}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_shape)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, {2, 3});
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call_with_validate(f, {a}, {c, b}));
}
