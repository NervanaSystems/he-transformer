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
    auto m_he_backend = runtime::Backend::create("HE");
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
    auto result = m_he_backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    EXPECT_ANY_THROW(m_he_backend->call(f, {result}, {a, b, c}));
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
