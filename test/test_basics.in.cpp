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

#include "gtest/gtest.h"
#include "he_backend.hpp"
#include "ngraph/log.hpp"
#include "test_main.hpp"

#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST_F(TestHEBackend, trivial)
{
    int a = 1;
    int b = 2;
    EXPECT_EQ(3, a + b);
}

TEST_F(TestHEBackend, backend_init)
{
    auto he_backend = runtime::Backend::create("HE");
    EXPECT_EQ(1, 1);
}

TEST_F(TestHEBackend, cipher_tv_write_read_scalar)
{
    Shape shape{};
    auto a = m_he_backend->create_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{5});
    EXPECT_EQ(read_vector<int64_t>(a), (vector<int64_t>{5}));
}

TEST_F(TestHEBackend, cipher_tv_write_read_2)
{
    Shape shape{2};
    auto a = m_he_backend->create_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{5, 6});
    EXPECT_EQ(read_vector<int64_t>(a), (vector<int64_t>{5, 6}));
}

TEST_F(TestHEBackend, cipher_tv_write_read_2_3)
{
    Shape shape{2, 3};
    auto a = m_he_backend->create_tensor(element::i64, shape);
    copy_data(a, test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_EQ(read_vector<int64_t>(a),
              (test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector());
}

TEST_F(TestHEBackend, plain_tv_write_read_scalar)
{
    Shape shape{};
    auto a = m_he_backend->create_plain_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{5});
    EXPECT_EQ(read_vector<int64_t>(a), (vector<int64_t>{5}));
}

TEST_F(TestHEBackend, plain_tv_write_read_2)
{
    Shape shape{2};
    auto a = m_he_backend->create_plain_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{5, 6});
    EXPECT_EQ(read_vector<int64_t>(a), (vector<int64_t>{5, 6}));
}

TEST_F(TestHEBackend, plain_tv_write_read_2_3)
{
    Shape shape{2, 3};
    auto a = m_he_backend->create_plain_tensor(element::i64, shape);
    copy_data(a, test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_EQ(read_vector<int64_t>(a),
              (test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector());
}

TEST_F(TestHEBackend, ab)
{
    Shape s{2, 3};
    auto a = make_shared<op::Parameter>(element::i64, s);
    auto b = make_shared<op::Parameter>(element::i64, s);
    auto t = make_shared<op::Add>(a, b);
    auto f = make_shared<Function>(t, op::ParameterVector{a, b});

    // Create some tensors for input/output
    auto t_a = m_he_backend->create_tensor(element::i64, s);
    auto t_b = m_he_backend->create_tensor(element::i64, s);
    auto t_result = m_he_backend->create_tensor(element::i64, s);

    copy_data(t_a, test::NDArray<int64_t, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
    copy_data(t_b, test::NDArray<int64_t, 2>({{7, 8, 9}, {10, 11, 12}}).get_vector());

    m_he_backend->call(f, {t_result}, {t_a, t_b});
    EXPECT_EQ(read_vector<int64_t>(t_result),
              (test::NDArray<int64_t, 2>({{8, 10, 12}, {14, 16, 18}})).get_vector());
}

TEST_F(TestHEBackend, ab_plain)
{
    Shape s{2, 3};
    auto a = make_shared<op::Parameter>(element::i64, s);
    auto b = make_shared<op::Parameter>(element::i64, s);
    auto t = make_shared<op::Add>(a, b);
    auto f = make_shared<Function>(t, op::ParameterVector{a, b});

    // Create some tensors for input/output
    auto t_a = m_he_backend->create_tensor(element::i64, s);
    auto t_b = m_he_backend->create_plain_tensor(element::i64, s);
    auto t_result = m_he_backend->create_tensor(element::i64, s);

    copy_data(t_a, test::NDArray<int64_t, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
    copy_data(t_b, test::NDArray<int64_t, 2>({{7, 8, 9}, {10, 11, 12}}).get_vector());

    m_he_backend->call(f, {t_result}, {t_a, t_b});
    EXPECT_EQ(read_vector<int64_t>(t_result),
              (test::NDArray<int64_t, 2>({{8, 10, 12}, {14, 16, 18}})).get_vector());
}

TEST_F(TestHEBackend, subtract)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Subtract>(A, B), op::ParameterVector{A, B});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = m_he_backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = m_he_backend->create_tensor(element::f32, shape);

    m_he_backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, 4, 8}), read_vector<float>(result));
}

TEST_F(TestHEBackend, subtract_plain)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Subtract>(A, B), op::ParameterVector{A, B});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = m_he_backend->create_plain_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = m_he_backend->create_tensor(element::f32, shape);

    m_he_backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, 4, 8}), read_vector<float>(result));
}

TEST_F(TestHEBackend, abc)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape);
    auto b = m_he_backend->create_tensor(element::f32, shape);
    auto c = m_he_backend->create_tensor(element::f32, shape);
    auto result = m_he_backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    m_he_backend->call(f, {result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    m_he_backend->call(f, {result}, {b, a, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    m_he_backend->call(f, {result}, {a, c, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());
}

// This test should be updated as the backend is able to handle deeper computation
TEST_F(TestHEBackend, abcd_budget)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto D = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(((A * B) * C) * D, op::ParameterVector{A, B, C, D});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape);
    auto b = m_he_backend->create_tensor(element::f32, shape);
    auto c = m_he_backend->create_tensor(element::f32, shape);
    auto d = m_he_backend->create_tensor(element::f32, shape);
    auto result = m_he_backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());
    copy_data(d, test::NDArray<float, 2>({{13, 14}, {15, 16}}).get_vector());

    //EXPECT_ANY_THROW(m_he_backend->call(f, {result}, {a, b, c, d}));
    m_he_backend->call(f, {result}, {a, b, c, d});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{585, 1680}, {3465, 6144}})).get_vector());
}

// This test should be updated as the backend is able to handle deeper computation
TEST_F(TestHEBackend, abcde_budget)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto D = make_shared<op::Parameter>(element::f32, shape);
    auto E = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((((A * B) * C) * D) * E, op::ParameterVector{A, B, C, D, E});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape);
    auto b = m_he_backend->create_tensor(element::f32, shape);
    auto c = m_he_backend->create_tensor(element::f32, shape);
    auto d = m_he_backend->create_tensor(element::f32, shape);
    auto e = m_he_backend->create_tensor(element::f32, shape);
    auto result = m_he_backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());
    copy_data(d, test::NDArray<float, 2>({{13, 14}, {15, 16}}).get_vector());
    copy_data(d, test::NDArray<float, 2>({{17, 18}, {19, 20}}).get_vector());

    EXPECT_ANY_THROW(m_he_backend->call(f, {result}, {a, b, c, d, e}));
    //m_he_backend->call(f, {result}, {a, b, c, d, e});
    //EXPECT_EQ(read_vector<float>(result),
    //        (test::NDArray<float, 2>({{585, 1680}, {3465, 6144}})).get_vector());
}

TEST_F(TestHEBackend, abc_budget)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A * B) * C, op::ParameterVector{A, B, C});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape);
    auto b = m_he_backend->create_tensor(element::f32, shape);
    auto c = m_he_backend->create_tensor(element::f32, shape);
    auto d = m_he_backend->create_tensor(element::f32, shape);
    auto result = m_he_backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    m_he_backend->call(f, {result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector());
}

TEST_F(TestHEBackend, abc_plain)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape);
    auto b = m_he_backend->create_plain_tensor(element::f32, shape);
    auto c = m_he_backend->create_plain_tensor(element::f32, shape);
    auto result = m_he_backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    m_he_backend->call(f, {result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    m_he_backend->call(f, {result}, {b, a, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    m_he_backend->call(f, {result}, {a, c, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());
}

TEST_F(TestHEBackend, add_float_precision)
{
    Shape s{};
    auto a = make_shared<op::Parameter>(element::f32, s);
    auto b = make_shared<op::Parameter>(element::f32, s);
    auto t = make_shared<op::Add>(a, b);
    auto f = make_shared<Function>(t, op::ParameterVector{a, b});

    // Create some tensors for input/output
    auto t_a = m_he_backend->create_tensor(element::f32, s);
    auto t_b = m_he_backend->create_tensor(element::f32, s);
    auto t_result = m_he_backend->create_tensor(element::f32, s);

    for (float sign = -1; sign <= 1; sign += 2)
    {
        for (float power = -30; power < 30; ++power)
        {
            copy_data(t_a, vector<float>{sign * 1 * powf(2, power)});
            copy_data(t_b, vector<float>{sign * 7 * powf(2, power)});

            m_he_backend->call(f, {t_result}, {t_a, t_b});
            EXPECT_EQ(read_vector<float>(t_result), vector<float>{sign * 8 * powf(2, power)});
        }
    }
}

TEST_F(TestHEBackend, add_int64_precision)
{
    Shape s{};
    auto a = make_shared<op::Parameter>(element::i64, s);
    auto b = make_shared<op::Parameter>(element::i64, s);
    auto t = make_shared<op::Add>(a, b);
    auto f = make_shared<Function>(t, op::ParameterVector{a, b});

    // Create some tensors for input/output
    auto t_a = m_he_backend->create_tensor(element::i64, s);
    auto t_b = m_he_backend->create_tensor(element::i64, s);
    auto t_result = m_he_backend->create_tensor(element::i64, s);

    for (int64_t sign = -1; sign <= 1; sign += 2)
    {
        for (int64_t power = 0; power < 30; ++power)
        {
            copy_data(t_a, vector<int64_t>{sign * 1 * (int64_t)pow(2, power)});
            copy_data(t_b, vector<int64_t>{sign * 7 * (int64_t)pow(2, power)});

            m_he_backend->call(f, {t_result}, {t_a, t_b});
            EXPECT_EQ(read_vector<int64_t>(t_result),
                      vector<int64_t>{sign * 8 * (int64_t)pow(2, power)});
        }
    }
}

TEST_F(TestHEBackend, mult_float_precision)
{
    Shape s{};
    auto a = make_shared<op::Parameter>(element::f32, s);
    auto b = make_shared<op::Parameter>(element::f32, s);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, op::ParameterVector{a, b});

    // Create some tensors for input/output
    auto t_a = m_he_backend->create_tensor(element::f32, s);
    auto t_b = m_he_backend->create_tensor(element::f32, s);
    auto t_result = m_he_backend->create_tensor(element::f32, s);

    for (float sign = -1; sign <= 1; sign += 2)
    {
        for (float power = -14; power < 30; ++power)
        {
            copy_data(t_a, vector<float>{sign * 2 * powf(2, power)});
            copy_data(t_b, vector<float>{3 * powf(2, power)});

            m_he_backend->call(f, {t_result}, {t_a, t_b});
            EXPECT_EQ(read_vector<float>(t_result), vector<float>{sign * 6 * powf(2, 2 * power)});
        }
    }
}

TEST_F(TestHEBackend, mult_int64_precision)
{
    Shape s{};
    auto a = make_shared<op::Parameter>(element::i64, s);
    auto b = make_shared<op::Parameter>(element::i64, s);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, op::ParameterVector{a, b});

    // Create some tensors for input/output
    auto t_a = m_he_backend->create_tensor(element::i64, s);
    auto t_b = m_he_backend->create_tensor(element::i64, s);
    auto t_result = m_he_backend->create_tensor(element::i64, s);

    for (int64_t sign = -1; sign <= 1; sign += 2)
    {
        for (int64_t power = 0; power < 30; ++power)
        {
            copy_data(t_a, vector<int64_t>{sign * 2 * (int64_t)pow(2, power)});
            copy_data(t_b, vector<int64_t>{3 * (int64_t)pow(2, power)});

            m_he_backend->call(f, {t_result}, {t_a, t_b});
            EXPECT_EQ(read_vector<int64_t>(t_result),
                      vector<int64_t>{sign * 6 * (int64_t)pow(2, 2 * power)});
        }
    }
}

TEST_F(TestHEBackend, dot1d)
{
    Shape shape{4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 4, 8});
    auto b = m_he_backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2, 4, 8, 16});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{170}), read_vector<float>(result));
}

TEST_F(TestHEBackend, dot1d_plain_binary)
{
    Shape shape{4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    // Create some tensors for input/output
    auto a = m_he_backend->create_plain_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, -1, -1});
    auto b = m_he_backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2, 4, 8, 16});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{-18}), read_vector<float>(result));
}

TEST_F(TestHEBackend, dot1d_plain)
{
    Shape shape{4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    // Create some tensors for input/output
    auto a = m_he_backend->create_plain_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 4, 8});
    auto b = m_he_backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2, 4, 8, 16});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{170}), read_vector<float>(result));
}

TEST_F(TestHEBackend, dot_matrix_vector)
{
    Shape shape_a{4, 4};
    Shape shape_b{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});
    Shape shape_r{4};

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = m_he_backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{17, 18, 19, 20});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{190, 486, 782, 1078}), read_vector<float>(result));
}

TEST_F(TestHEBackend, dot_matrix_vector_plain)
{
    Shape shape_a{4, 4};
    Shape shape_b{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});
    Shape shape_r{4};

    // Create some tensors for input/output
    auto a = m_he_backend->create_plain_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = m_he_backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{17, 18, 19, 20});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{190, 486, 782, 1078}), read_vector<float>(result));
}

TEST_F(TestHEBackend, dot_scalar_scalar)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{8});
    auto b = m_he_backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{6});
    auto result = m_he_backend->create_tensor(element::f32, shape);

    m_he_backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{48}), read_vector<float>(result));
}

TEST_F(TestHEBackend, dot_scalar_scalar_plain)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    // Create some tensors for input/output
    auto a = m_he_backend->create_plain_tensor(element::f32, shape);
    copy_data(a, vector<float>{8});
    auto b = m_he_backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{6});
    auto result = m_he_backend->create_tensor(element::f32, shape);

    m_he_backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{48}), read_vector<float>(result));
}

TEST_F(TestHEBackend, constant)
{
    Shape shape{2, 2};
    auto A = op::Constant::create(element::f32, shape, {0.1, 0.2, 0.3, 0.4});
    auto f = make_shared<Function>(A, op::ParameterVector{});

    auto result = m_he_backend->create_tensor(element::f32, shape);
    m_he_backend->call(f, {result}, {});
    EXPECT_EQ((vector<float>{0.1, 0.2, 0.3, 0.4}), read_vector<float>(result));
}

TEST_F(TestHEBackend, constant_abc)
{
    Shape shape{2, 2};
    auto A = op::Constant::create(element::f32, shape, {1, 2, 3, 4});
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{B, C});

    // Create some tensors for input/output
    auto b = m_he_backend->create_tensor(element::f32, shape);
    auto c = m_he_backend->create_tensor(element::f32, shape);
    auto result = m_he_backend->create_tensor(element::f32, shape);

    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    m_he_backend->call(f, {result}, {b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());
}

TEST_F(TestHEBackend, broadcast_scalar_vector)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{6, 6, 6, 6}), read_vector<float>(result));
}

TEST_F(TestHEBackend, broadcast_scalar_vector_plain)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_plain_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{6, 6, 6, 6}), read_vector<float>(result));
}

TEST_F(TestHEBackend, broadcast_to_non_existent_axis)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{4};
    ASSERT_THROW(auto f = make_shared<Function>(
                     make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1}), op::ParameterVector{A}),
                 ngraph_error);
}

TEST_F(TestHEBackend, broadcast_scalar_matrix)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1}),
                                   op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{6, 6, 6, 6}), read_vector<float>(result));
}

TEST_F(TestHEBackend, broadcast_scalar_tensor)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1, 2}),
                                   op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{6, 6, 6, 6, 6, 6, 6, 6}), read_vector<float>(result));
}

TEST_F(TestHEBackend, broadcast_trivial)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape, AxisSet{}),
                                   op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 6, 8, 16, 32, 64, 128});
    auto result = m_he_backend->create_tensor(element::f32, shape);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{2, 4, 6, 8, 16, 32, 64, 128}), read_vector<float>(result));
}

TEST_F(TestHEBackend, broadcast_vector_colwise)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{1}),
                                   op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}), read_vector<float>(result));
}

TEST_F(TestHEBackend, broadcast_vector_rowwise)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}), read_vector<float>(result));
}

TEST_F(TestHEBackend, broadcast_vector_rowwise_int64)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_r{3, 4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{1, 2, 3, 4});
    auto result = m_he_backend->create_tensor(element::i64, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<int64_t>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}), read_vector<int64_t>(result));
}

TEST_F(TestHEBackend, broadcast_matrix_0)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 1, 2, 3, 4}), read_vector<float>(result));
}

TEST_F(TestHEBackend, broadcast_matrix_1)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{1}),
                                   op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 1, 2, 3, 4, 3, 4}), read_vector<float>(result));
}

TEST_F(TestHEBackend, broadcast_matrix_2)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{2}),
                                   op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 2, 2, 3, 3, 4, 4}), read_vector<float>(result));
}

TEST_F(TestHEBackend, reshape_t2v_012)
{
    Shape shape_a{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{12};
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), read_vector<float>(result));
}

TEST_F(TestHEBackend, reshape_t2v_012_plain)
{
    Shape shape_a{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{12};
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_plain_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), read_vector<float>(result));
}

TEST_F(TestHEBackend, reshape_t2s_012)
{
    Shape shape_a{1, 1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{};
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{6}), read_vector<float>(result));
}

TEST_F(TestHEBackend, reshape_t2s_120)
{
    Shape shape_a{1, 1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{};
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 2, 0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{6}), read_vector<float>(result));
}

TEST_F(TestHEBackend, reshape_s2t)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 1, 1, 1, 1};
    auto r = make_shared<op::Reshape>(A, AxisVector{}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{42});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{42}), read_vector<float>(result));
}

TEST_F(TestHEBackend, reshape_v2m_col)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 1};
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result));
}

TEST_F(TestHEBackend, reshape_v2m_row)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 3};
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result));
}

TEST_F(TestHEBackend, reshape_v2t_middle)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 3, 1};
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result));
}

TEST_F(TestHEBackend, reshape_m2m_same)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 3};
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), read_vector<float>(result));
}

TEST_F(TestHEBackend, reshape_m2m_transpose)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 3};
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 4, 7, 2, 5, 8, 3, 6, 9}), read_vector<float>(result));
}

TEST_F(TestHEBackend, reshape_m2m_dim_change_transpose)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 3};
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 3, 5, 2, 4, 6}), read_vector<float>(result));
}

//
// Numpy:
//
// >>> x = linspace(1,2*2*3*3*2*4,2*2*3*3*2*4)
// >>> x.shape=(2,2,3,3,2,4)
// >>> y = ascontiguousarray(transpose(x,(2,4,0,5,3,1)))
// >>> y.shape=2*2*3*3*2*4
// >>> y
// array([   1.,   73.,    9.,   81.,   17.,   89.,    2.,   74.,   10.,
//          82.,   18.,   90.,    3.,   75.,   11.,   83.,   19.,   91.,
//           4.,   76.,   12.,   84.,   20.,   92.,  145.,  217.,  153.,
//         225.,  161.,  233.,  146.,  218.,  154.,  226.,  162.,  234.,
//         147.,  219.,  155.,  227.,  163.,  235.,  148.,  220.,  156.,
//         228.,  164.,  236.,    5.,   77.,   13.,   85.,   21.,   93.,
//           6.,   78.,   14.,   86.,   22.,   94.,    7.,   79.,   15.,
//          87.,   23.,   95.,    8.,   80.,   16.,   88.,   24.,   96.,
//         149.,  221.,  157.,  229.,  165.,  237.,  150.,  222.,  158.,
//         230.,  166.,  238.,  151.,  223.,  159.,  231.,  167.,  239.,
//         152.,  224.,  160.,  232.,  168.,  240.,   25.,   97.,   33.,
//         105.,   41.,  113.,   26.,   98.,   34.,  106.,   42.,  114.,
//          27.,   99.,   35.,  107.,   43.,  115.,   28.,  100.,   36.,
//         108.,   44.,  116.,  169.,  241.,  177.,  249.,  185.,  257.,
//         170.,  242.,  178.,  250.,  186.,  258.,  171.,  243.,  179.,
//         251.,  187.,  259.,  172.,  244.,  180.,  252.,  188.,  260.,
//          29.,  101.,   37.,  109.,   45.,  117.,   30.,  102.,   38.,
//         110.,   46.,  118.,   31.,  103.,   39.,  111.,   47.,  119.,
//          32.,  104.,   40.,  112.,   48.,  120.,  173.,  245.,  181.,
//         253.,  189.,  261.,  174.,  246.,  182.,  254.,  190.,  262.,
//         175.,  247.,  183.,  255.,  191.,  263.,  176.,  248.,  184.,
//         256.,  192.,  264.,   49.,  121.,   57.,  129.,   65.,  137.,
//          50.,  122.,   58.,  130.,   66.,  138.,   51.,  123.,   59.,
//         131.,   67.,  139.,   52.,  124.,   60.,  132.,   68.,  140.,
//         193.,  265.,  201.,  273.,  209.,  281.,  194.,  266.,  202.,
//         274.,  210.,  282.,  195.,  267.,  203.,  275.,  211.,  283.,
//         196.,  268.,  204.,  276.,  212.,  284.,   53.,  125.,   61.,
//         133.,   69.,  141.,   54.,  126.,   62.,  134.,   70.,  142.,
//          55.,  127.,   63.,  135.,   71.,  143.,   56.,  128.,   64.,
//         136.,   72.,  144.,  197.,  269.,  205.,  277.,  213.,  285.,
//         198.,  270.,  206.,  278.,  214.,  286.,  199.,  271.,  207.,
//         279.,  215.,  287.,  200.,  272.,  208.,  280.,  216.,  288.])
//
TEST_F(TestHEBackend, reshape_6d)
{
    vector<float> a_data(2 * 2 * 3 * 3 * 2 * 4);
    for (int i = 0; i < 2 * 2 * 3 * 3 * 2 * 4; i++)
    {
        a_data[i] = float(i + 1);
    }

    Shape shape_a{2, 2, 3, 3, 2, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 2, 2, 4, 3, 2};

    auto r = make_shared<op::Reshape>(A, AxisVector{2, 4, 0, 5, 3, 1}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);

    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ(
        (vector<float>{
            1.,   73.,  9.,   81.,  17.,  89.,  2.,   74.,  10.,  82.,  18.,  90.,  3.,   75.,
            11.,  83.,  19.,  91.,  4.,   76.,  12.,  84.,  20.,  92.,  145., 217., 153., 225.,
            161., 233., 146., 218., 154., 226., 162., 234., 147., 219., 155., 227., 163., 235.,
            148., 220., 156., 228., 164., 236., 5.,   77.,  13.,  85.,  21.,  93.,  6.,   78.,
            14.,  86.,  22.,  94.,  7.,   79.,  15.,  87.,  23.,  95.,  8.,   80.,  16.,  88.,
            24.,  96.,  149., 221., 157., 229., 165., 237., 150., 222., 158., 230., 166., 238.,
            151., 223., 159., 231., 167., 239., 152., 224., 160., 232., 168., 240., 25.,  97.,
            33.,  105., 41.,  113., 26.,  98.,  34.,  106., 42.,  114., 27.,  99.,  35.,  107.,
            43.,  115., 28.,  100., 36.,  108., 44.,  116., 169., 241., 177., 249., 185., 257.,
            170., 242., 178., 250., 186., 258., 171., 243., 179., 251., 187., 259., 172., 244.,
            180., 252., 188., 260., 29.,  101., 37.,  109., 45.,  117., 30.,  102., 38.,  110.,
            46.,  118., 31.,  103., 39.,  111., 47.,  119., 32.,  104., 40.,  112., 48.,  120.,
            173., 245., 181., 253., 189., 261., 174., 246., 182., 254., 190., 262., 175., 247.,
            183., 255., 191., 263., 176., 248., 184., 256., 192., 264., 49.,  121., 57.,  129.,
            65.,  137., 50.,  122., 58.,  130., 66.,  138., 51.,  123., 59.,  131., 67.,  139.,
            52.,  124., 60.,  132., 68.,  140., 193., 265., 201., 273., 209., 281., 194., 266.,
            202., 274., 210., 282., 195., 267., 203., 275., 211., 283., 196., 268., 204., 276.,
            212., 284., 53.,  125., 61.,  133., 69.,  141., 54.,  126., 62.,  134., 70.,  142.,
            55.,  127., 63.,  135., 71.,  143., 56.,  128., 64.,  136., 72.,  144., 197., 269.,
            205., 277., 213., 285., 198., 270., 206., 278., 214., 286., 199., 271., 207., 279.,
            215., 287., 200., 272., 208., 280., 216., 288.}),
        read_vector<float>(result));
}

TEST_F(TestHEBackend, one_hot_scalar_2_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::i64, shape_a);
    NGRAPH_INFO << "created tensor, copying";
    copy_data(a, vector<int64_t>{2});
    auto result = m_he_backend->create_tensor(element::i64, shape_r);
    NGRAPH_INFO << "calling ";

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<int64_t>{0, 0, 1}), read_vector<int64_t>(result));
}

TEST_F(TestHEBackend, one_hot_scalar_1_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{1});
    auto result = m_he_backend->create_tensor(element::i64, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<int64_t>{0, 1, 0}), read_vector<int64_t>(result));
}

TEST_F(TestHEBackend, one_hot_scalar_0_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{0});
    auto result = m_he_backend->create_tensor(element::i64, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<int64_t>{1, 0, 0}), read_vector<int64_t>(result));
}

TEST_F(TestHEBackend, one_hot_scalar_fp_nonint_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1.1f});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    try
    {
        m_he_backend->call(f, {result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(
            e.what(),
            std::string("One-hot: non-integral value in input or value is out of category range"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

TEST_F(TestHEBackend, one_hot_scalar_oob_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{3000000});
    auto result = m_he_backend->create_tensor(element::i64, shape_r);

    try
    {
        m_he_backend->call(f, {result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(
            e.what(),
            std::string("One-hot: non-integral value in input or value is out of category range"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

TEST_F(TestHEBackend, one_hot_vector_0)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_r{3, 8};
    auto r = make_shared<op::OneHot>(A, Shape{3, 8}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = m_he_backend->create_tensor(element::i64, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ(
        (vector<int64_t>{0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0}),
        read_vector<int64_t>(result));
}

TEST_F(TestHEBackend, one_hot_vector_1)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = m_he_backend->create_tensor(element::i64, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ(
        (vector<int64_t>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0}),
        read_vector<int64_t>(result));
}

TEST_F(TestHEBackend, one_hot_vector_1_barely_oob)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{2, 1, 0, 0, 3, 2, 1, 0});
    auto result = m_he_backend->create_tensor(element::i64, shape_r);

    try
    {
        m_he_backend->call(f, {result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(
            e.what(),
            std::string("One-hot: non-integral value in input or value is out of category range"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}
TEST_F(TestHEBackend, one_hot_vector_1_far_oob)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{2, 1, 0, 0, 3000000, 2, 1, 0});
    auto result = m_he_backend->create_tensor(element::i64, shape_r);

    try
    {
        m_he_backend->call(f, {result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(
            e.what(),
            std::string("One-hot: non-integral value in input or value is out of category range"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

TEST_F(TestHEBackend, one_hot_matrix_0)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_r{3, 3, 3};
    auto r = make_shared<op::OneHot>(A, Shape{3, 3, 3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::i64, shape_a);
    copy_data(a,
              vector<int64_t>{
                  0, 1, 1, 2, 1, 0, 0, 2, 1,
              });
    auto result = m_he_backend->create_tensor(element::i64, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<int64_t>{1, 0, 0, 0, 0, 1, 1, 0, 0,

                               0, 1, 1, 0, 1, 0, 0, 0, 1,

                               0, 0, 0, 1, 0, 0, 0, 1, 0}),
              read_vector<int64_t>(result));
}

TEST_F(TestHEBackend, one_hot_vector_1_fp)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ(
        (vector<float>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0}),
        read_vector<float>(result));
}
TEST_F(TestHEBackend, one_hot_vector_1_fp_nonint)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 1, 0, 0, 2, 2, 1.01f, 0});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    try
    {
        m_he_backend->call(f, {result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(
            e.what(),
            std::string("One-hot: non-integral value in input or value is out of category range"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

TEST_F(TestHEBackend, slice_scalar)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{};
    auto r = make_shared<op::Slice>(A, Coordinate{}, Coordinate{});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{312});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{312}), read_vector<float>(result));
}

TEST_F(TestHEBackend, slice_matrix)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 1}, Coordinate{3, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{2, 3, 6, 7, 10, 11}), read_vector<float>(result));
}

TEST_F(TestHEBackend, slice_vector)
{
    Shape shape_a{16};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{12};
    auto r = make_shared<op::Slice>(A, Coordinate{2}, Coordinate{14});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}), read_vector<float>(result));
}

TEST_F(TestHEBackend, slice_matrix_strided)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{1, 0}, Coordinate{4, 4}, Strides{2, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{4, 7, 12, 15}), read_vector<float>(result));
}

TEST_F(TestHEBackend, slice_3d)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{1, 1, 1}, Coordinate{3, 3, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{21, 22, 25, 26, 37, 38, 41, 42}), read_vector<float>(result));
}

TEST_F(TestHEBackend, slice_3d_strided)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 2});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{0, 2, 8, 10, 32, 34, 40, 42}), read_vector<float>(result));
}

TEST_F(TestHEBackend, slice_3d_strided_different_strides)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{0, 3, 8, 11, 32, 35, 40, 43}), read_vector<float>(result));
}

TEST_F(TestHEBackend, concat_matrix_colwise)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 3};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 3};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 8};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 1),
                                   op::ParameterVector{A, B, C});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = m_he_backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = m_he_backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}),
              read_vector<float>(result));
}

TEST_F(TestHEBackend, concat_matrix_rowwise)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{3, 2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{8, 2};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   op::ParameterVector{A, B, C});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = m_he_backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = m_he_backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
              read_vector<float>(result));
}

TEST_F(TestHEBackend, concat_matrix_int64)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::i64, shape_b);
    Shape shape_c{3, 2};
    auto C = make_shared<op::Parameter>(element::i64, shape_c);
    Shape shape_r{8, 2};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   op::ParameterVector{A, B, C});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{2, 4, 8, 16});
    auto b = m_he_backend->create_tensor(element::i64, shape_b);
    copy_data(b, vector<int64_t>{1, 2, 4, 8, 16, 32});
    auto c = m_he_backend->create_tensor(element::i64, shape_c);
    copy_data(c, vector<int64_t>{2, 3, 5, 7, 11, 13});
    auto result = m_he_backend->create_tensor(element::i64, shape_r);

    m_he_backend->call(f, {result}, {a, b, c});
    EXPECT_EQ((vector<int64_t>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
              read_vector<int64_t>(result));
}

TEST_F(TestHEBackend, concat_vector)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{6};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{12};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   op::ParameterVector{A, B, C});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = m_he_backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = m_he_backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{18, 19});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}), read_vector<float>(result));
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
TEST_F(TestHEBackend, concat_5d)
{
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

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = m_he_backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);
    auto c = m_he_backend->create_tensor(element::f32, shape_c);
    copy_data(c, c_data);

    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a, b, c});
    EXPECT_EQ(
        (vector<float>{
            1.,    2.,    3.,    4.,    5.,    6.,    7.,    8.,    9.,    10.,   11.,   12.,
            13.,   14.,   15.,   16.,   17.,   18.,   19.,   20.,   21.,   22.,   23.,   24.,
            1001., 1002., 1003., 1004., 1005., 1006., 1007., 1008., 1009., 1010., 1011., 1012.,
            1013., 1014., 1015., 1016., 1017., 1018., 2001., 2002., 2003., 2004., 2005., 2006.,
            2007., 2008., 2009., 2010., 2011., 2012., 25.,   26.,   27.,   28.,   29.,   30.,
            31.,   32.,   33.,   34.,   35.,   36.,   37.,   38.,   39.,   40.,   41.,   42.,
            43.,   44.,   45.,   46.,   47.,   48.,   1019., 1020., 1021., 1022., 1023., 1024.,
            1025., 1026., 1027., 1028., 1029., 1030., 1031., 1032., 1033., 1034., 1035., 1036.,
            2013., 2014., 2015., 2016., 2017., 2018., 2019., 2020., 2021., 2022., 2023., 2024.,
            49.,   50.,   51.,   52.,   53.,   54.,   55.,   56.,   57.,   58.,   59.,   60.,
            61.,   62.,   63.,   64.,   65.,   66.,   67.,   68.,   69.,   70.,   71.,   72.,
            1037., 1038., 1039., 1040., 1041., 1042., 1043., 1044., 1045., 1046., 1047., 1048.,
            1049., 1050., 1051., 1052., 1053., 1054., 2025., 2026., 2027., 2028., 2029., 2030.,
            2031., 2032., 2033., 2034., 2035., 2036., 73.,   74.,   75.,   76.,   77.,   78.,
            79.,   80.,   81.,   82.,   83.,   84.,   85.,   86.,   87.,   88.,   89.,   90.,
            91.,   92.,   93.,   94.,   95.,   96.,   1055., 1056., 1057., 1058., 1059., 1060.,
            1061., 1062., 1063., 1064., 1065., 1066., 1067., 1068., 1069., 1070., 1071., 1072.,
            2037., 2038., 2039., 2040., 2041., 2042., 2043., 2044., 2045., 2046., 2047., 2048.,
            97.,   98.,   99.,   100.,  101.,  102.,  103.,  104.,  105.,  106.,  107.,  108.,
            109.,  110.,  111.,  112.,  113.,  114.,  115.,  116.,  117.,  118.,  119.,  120.,
            1073., 1074., 1075., 1076., 1077., 1078., 1079., 1080., 1081., 1082., 1083., 1084.,
            1085., 1086., 1087., 1088., 1089., 1090., 2049., 2050., 2051., 2052., 2053., 2054.,
            2055., 2056., 2057., 2058., 2059., 2060., 121.,  122.,  123.,  124.,  125.,  126.,
            127.,  128.,  129.,  130.,  131.,  132.,  133.,  134.,  135.,  136.,  137.,  138.,
            139.,  140.,  141.,  142.,  143.,  144.,  1091., 1092., 1093., 1094., 1095., 1096.,
            1097., 1098., 1099., 1100., 1101., 1102., 1103., 1104., 1105., 1106., 1107., 1108.,
            2061., 2062., 2063., 2064., 2065., 2066., 2067., 2068., 2069., 2070., 2071., 2072.}),
        read_vector<float>(result));
}
