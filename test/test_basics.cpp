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
#include "util/test_tools.hpp"

#include "he_backend.hpp"
#include "test_util.hpp"

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

TEST_F(TestHEBackend, seal_debug)
{
#ifndef SEAL_DEBUG
    EXPECT_EQ(1, 2);
#endif
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

TEST_F(TestHEBackend, ab_plain_plain)
{
    Shape s{2, 3};
    auto a = make_shared<op::Parameter>(element::f32, s);
    auto b = make_shared<op::Parameter>(element::f32, s);
    auto t = make_shared<op::Add>(a, b);
    auto f = make_shared<Function>(t, op::ParameterVector{a, b});

    // Create some tensors for input/output
    auto t_a = m_he_backend->create_plain_tensor(element::f32, s);
    auto t_b = m_he_backend->create_plain_tensor(element::f32, s);
    auto t_result = m_he_backend->create_plain_tensor(element::f32, s);

    copy_data(t_a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
    copy_data(t_b, test::NDArray<float, 2>({{7, 8, 9}, {10, 11, 12}}).get_vector());

    m_he_backend->call(f, {t_result}, {t_a, t_b});
    EXPECT_EQ(read_vector<float>(t_result),
              (test::NDArray<float, 2>({{8, 10, 12}, {14, 16, 18}})).get_vector());
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

TEST_F(TestHEBackend, abc_plain_plain)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    // Create some tensors for input/output
    auto a = m_he_backend->create_plain_tensor(element::f32, shape);
    auto b = m_he_backend->create_plain_tensor(element::f32, shape);
    auto c = m_he_backend->create_plain_tensor(element::f32, shape);
    auto result = m_he_backend->create_plain_tensor(element::f32, shape);

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

TEST_F(TestHEBackend, dot1d_plain_plain)
{
    Shape shape{4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    // Create some tensors for input/output
    auto a = m_he_backend->create_plain_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 4, 8});
    auto b = m_he_backend->create_plain_tensor(element::f32, shape);
    copy_data(b, vector<float>{2, 4, 8, 16});
    auto result = m_he_backend->create_plain_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{170}), read_vector<float>(result));
}

TEST_F(TestHEBackend, dot1d_plain_binary)
{
    Shape shape{16};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    // Create some tensors for input/output
    auto a = m_he_backend->create_plain_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1});
    auto b = m_he_backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto result = m_he_backend->create_tensor(element::f32, shape_r);

    m_he_backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{-64}), read_vector<float>(result));
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

TEST_F(TestHEBackend, reshape_t2v_012_plain_plain)
{
    Shape shape_a{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{12};
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_plain_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto result = m_he_backend->create_plain_tensor(element::f32, shape_r);

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
        EXPECT_EQ(e.what(), std::string("One-hot: non-integral value in input"));
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
        EXPECT_EQ(e.what(), std::string("One-hot: value is out of category range"));
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
        EXPECT_EQ(e.what(), std::string("One-hot: value is out of category range"));
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
        EXPECT_EQ(e.what(), std::string("One-hot: value is out of category range"));
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
        EXPECT_EQ(e.what(), std::string("One-hot: non-integral value in input"));
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

// Trivial case with no summed axes.
TEST_F(TestHEBackend, sum_trivial)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = m_he_backend->create_tensor(element::f32, shape);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

// Failure has been reported at 5D for some reason
TEST_F(TestHEBackend, sum_trivial_5d)
{
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = m_he_backend->create_tensor(element::f32, shape);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<float>(result));
}

TEST_F(TestHEBackend, sum_to_scalar)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = m_he_backend->create_tensor(element::f32, Shape{});

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{10}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
}

TEST_F(TestHEBackend, sum_matrix_columns)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = m_he_backend->create_tensor(element::f32, shape_rt);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{9, 12}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

TEST_F(TestHEBackend, sum_matrix_rows)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = m_he_backend->create_tensor(element::f32, shape_rt);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{3, 7, 11}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

TEST_F(TestHEBackend, sum_matrix_rows_zero)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = m_he_backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{0, 0, 0}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST_F(TestHEBackend, sum_matrix_cols_zero)
{
    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = m_he_backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{0, 0}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST_F(TestHEBackend, sum_vector_zero)
{
    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = m_he_backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{0}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST_F(TestHEBackend, sum_matrix_to_scalar_zero_by_zero)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = m_he_backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{0}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST_F(TestHEBackend, sum_3d_to_matrix_most_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = m_he_backend->create_tensor(element::f32, shape_rt);

    m_he_backend->call(f, {result}, {a});
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

TEST_F(TestHEBackend, sum_3d_to_matrix_least_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{2}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = m_he_backend->create_tensor(element::f32, shape_rt);

    m_he_backend->call(f, {result}, {a});
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
TEST_F(TestHEBackend, sum_3d_to_vector)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = m_he_backend->create_tensor(element::f32, shape_rt);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25,
                             2 + 11 + 20 + 5 + 14 + 23 + 8 + 17 + 26,
                             3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
              read_vector<float>(result));
}

TEST_F(TestHEBackend, sum_3d_to_scalar)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f =
        make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1, 2}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = m_he_backend->create_tensor(element::f32, shape_rt);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25 + 2 + 11 + 20 + 5 + 14 + 23 +
                             8 + 17 + 26 + 3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
              read_vector<float>(result));
}

TEST_F(TestHEBackend, sum_3d_eliminate_zero_dim)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = m_he_backend->create_tensor(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{0, 0, 0, 0, 0, 0}), read_vector<float>(result));
}

TEST_F(TestHEBackend, sum_to_scalar_stable)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1e-6f, -1, 0, 1});
    auto result = m_he_backend->create_tensor(element::f32, Shape{});

    m_he_backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(read_vector<float>(result), vector<float>{1e-6f}, 5e-2f));
    // EXPECT_EQ(vector<float>{1e-6}, read_vector<float>(result));
}

TEST_F(TestHEBackend, sum_3d_to_vector_stable)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 1,  1,  1,  1,  1,  1e-4f, 1e-5f, 1e-6f, 1,  1,  1,  1, 1,
                               1, -1, -1, -1, -1, -1, -1,    -1,    -1,    -1, -1, -1, -1});
    auto result = m_he_backend->create_tensor(element::f32, shape_rt);

    m_he_backend->call(f, {result}, {a});
    EXPECT_TRUE(
        test::all_close(read_vector<float>(result), vector<float>{1e-4f, 1e-5f, 1e-6f}, 5e-2f));
}

TEST_F(TestHEBackend, sum_5d_to_scalar)
{
    Shape shape_a{3, 3, 3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1, 2, 3, 4}),
                                   op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(a, std::vector<float>(std::pow(3, 5), 1));
    auto result = m_he_backend->create_tensor(element::f32, shape_rt);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ(std::vector<float>{243.}, read_vector<float>(result));
}

TEST_F(TestHEBackend, sum_100)
{
    Shape shape_a{100, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(
        a,
        vector<float>{
            14.2377,   17.9675,   -1.27236, 0.746558,  4.31638,   -1.63154,  7.57737,   8.62198,
            9.05407,   17.8358,   4.4884,   7.73092,   1.0455,    -0.172748, 9.99637,   11.6235,
            6.48216,   7.39645,   7.6241,   12.8839,   4.94323,   6.87503,   -7.30094,  -13.7126,
            -0.156084, -5.64857,  8.0253,   6.98431,   7.02197,   5.18864,   -3.17431,  -6.38038,
            -0.349455, -2.58513,  3.15115,  3.23586,   4.2416,    10.899,    -17.532,   -15.6202,
            10.5308,   17.5596,   -2.27113, 1.59372,   2.95618,   1.96892,   9.01785,   3.6869,
            7.67656,   16.4888,   7.8306,   4.3149,    -1.23714,  -3.8089,   4.47179,   7.95677,
            8.61931,   13.2621,   5.78856,  0.892958,  0.0120182, -9.43983,  5.14599,   7.44762,
            -2.41991,  -4.61807,  14.2163,  11.6886,   4.77798,   6.22966,   8.67271,   8.25992,
            11.0431,   11.668,    6.84985,  -1.42819,  -1.71882,  -4.56618,  5.65461,   5.41699,
            -4.92285,  -7.0546,   13.7121,  4.28443,   7.52616,   7.76641,   11.2756,   5.72957,
            0.416175,  -0.549351, 2.20854,  4.63625,   -2.03635,  1.68677,   7.50243,   8.74035,
            -0.395208, 2.86464,   -2.65065, -8.44581,  4.34911,   -0.758135, 1.02894,   -0.177512,
            4.70663,   9.92955,   1.46842,  -2.62967,  8.55083,   14.6971,   4.05932,   5.19494,
            4.13868,   10.567,    3.83127,  6.39482,   6.71121,   7.32437,   2.28241,   1.80384,
            1.31484,   5.82456,   7.70988,  10.7845,   6.72938,   11.2826,   -0.486553, -3.23754,
            6.25243,   14.0468,   2.31849,  11.4065,   2.02936,   -2.73059,  2.77202,   -1.6188,
            -9.54767,  -8.46987,  -3.72039, -0.663913, 6.14345,   9.7325,    13.4762,   11.8298,
            7.78562,   10.0277,   7.3408,   12.6809,   5.16275,   5.89017,   4.01913,   8.79178,
            6.56057,   4.711,     9.48108,  3.06343,   5.92353,   9.66764,   -7.32664,  -4.91296,
            6.8446,    4.55041,   5.45017,  3.69992,   4.12889,   6.44041,   6.41508,   6.95436,
            5.37921,   16.5279,   3.16765,  7.54378,   -2.29099,  -3.74811,  9.40147,   6.67805,
            0.152946,  -6.16828,  8.13558,  8.96376,   3.54844,   12.7023,   0.664185,  2.52428,
            3.56561,   4.45346,   13.321,   22.5078,   12.9974,   18.2641,   6.50662,   20.7831,
            5.85657,   9.34474,   4.17526,  14.3107,   -11.2853,  -15.6588,  2.35086,   -0.901023});
    auto result = m_he_backend->create_tensor(element::f32, shape_rt);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{404.2236542, 486.696594}), read_vector<float>(result));
}

TEST_F(TestHEBackend, sum_100_2)
{
    Shape shape_a{100, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), op::ParameterVector{A});

    // Create some tensors for input/output
    auto a = m_he_backend->create_tensor(element::f32, shape_a);
    copy_data(
        a,
        vector<float>{
            4.68672,  44.2453,  -15.5205, -17.8839,  33.73,      -70.4566, -9.78681,  20.9547,
            1.9905,   49.7646,  18.0394,  13.2293,   32.3171,    47.2801,  18.6513,   49.5794,
            9.45922,  39.0961,  19.9755,  34.6886,   32.5691,    28.6711,  11.6226,   3.95575,
            12.4684,  -2.49802, 5.24948,  36.2466,   27.7646,    42.4754,  39.9205,   33.1619,
            -5.8029,  -5.73632, 15.9504,  18.2661,   -6.82032,   34.3888,  -33.9594,  -25.9467,
            -1.54458, 26.7049,  20.6567,  43.1609,   29.5706,    51.5047,  -0.795216, 32.2638,
            33.2262,  21.6088,  6.83814,  14.4333,   3.00686,    14.5201,  11.5473,   11.0324,
            -4.09393, 36.1957,  1.62459,  22.3159,   -3.41694,   -6.42114, 13.4633,   -7.22597,
            5.38151,  50.4182,  8.48233,  45.6641,   16.0715,    42.8035,  32.5209,   48.4259,
            4.27237,  -30.0045, 34.2212,  31.3201,   -0.0681592, 14.4702,  25.12,     54.576,
            11.0355,  31.0627,  32.0611,  40.6549,   20.4167,    33.7035,  -7.10309,  -29.2895,
            1.57968,  9.79366,  -10.1292, -15.9869,  6.41041,    -2.32292, 22.9415,   43.2013,
            36.202,   5.52278,  -6.70877, 8.5963,    33.7541,    19.3275,  29.1208,   9.57898,
            -6.71591, -2.7228,  27.9362,  25.6672,   -0.720666,  20.1775,  -12.6515,  35.6895,
            5.63928,  -11.1021, 9.53794,  14.629,    10.3581,    13.9774,  0.367201,  43.9667,
            9.36125,  -9.2531,  -18.6628, 35.1217,   6.15054,    16.8491,  12.6205,   47.4086,
            15.9988,  8.18875,  25.9422,  35.0453,   -6.99174,   25.4751,  37.4208,   21.2361,
            10.1736,  21.6833,  7.23031,  7.09738,   0.50878,    21.6973,  -0.683491, 48.4339,
            -7.19123, 38.8538,  -10.2643, 22.674,    -30.9213,   -41.7096, 7.80512,   27.9744,
            15.5493,  -9.76069, 25.165,   22.7478,   39.349,     28.8581,  -2.96684,  5.72495,
            17.0429,  6.17896,  35.8259,  10.645,    -11.1896,   15.4263,  38.2692,   67.7108,
            10.8544,  -4.16798, 14.3838,  3.72914,   5.15611,    -13.988,  -10.1345,  6.12128,
            -13.6677, -6.6537,  -4.45775, 45.4407,   3.24835,    37.3684,  -0.765132, 6.88647,
            5.10294,  7.29367,  28.1834,  53.2004,   14.4721,    16.9072,  0.592263,  15.7312,
            12.3032,  13.0224,  -9.30238, -0.913696, -5.44807,   -16.4989, 25.8421,   30.0779});
    auto result = m_he_backend->create_tensor(element::f32, shape_rt);

    m_he_backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{943.8259698, 1853.237534}), read_vector<float>(result));
}

/* TEST_F(TestHEBackend, create_valued_plaintext)
{
    // Fractional
    {
        float val = 3.14;
        element::Type type = element::f32;
        seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::New(false);
        shared_ptr<seal::Plaintext> plaintext =
            m_he_backend->create_valued_plaintext(val, type, pool);
        float val_decoded;
        m_he_backend->decode(&val_decoded, *plaintext, type);
        EXPECT_EQ(val_decoded, val);
    }

    // Integer
    {
        int64_t val = 1;
        element::Type type = element::i64;
        seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::New(false);
        shared_ptr<seal::Plaintext> plaintext =
            m_he_backend->create_valued_plaintext((float)val, type, pool);
        int64_t val_decoded;
        m_he_backend->decode(&val_decoded, *plaintext, type);
        EXPECT_EQ(val_decoded, val);
    }
    {
        int64_t val = 0;
        element::Type type = element::i64;
        seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::New(false);
        shared_ptr<seal::Plaintext> plaintext =
            m_he_backend->create_valued_plaintext((float)val, type, pool);
        int64_t val_decoded;
        m_he_backend->decode(&val_decoded, *plaintext, type);
        EXPECT_EQ(val_decoded, val);
    }
    {
        int64_t val = -2;
        element::Type type = element::i64;
        seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::New(false);
        shared_ptr<seal::Plaintext> plaintext =
            m_he_backend->create_valued_plaintext((float)val, type, pool);
        int64_t val_decoded;
        m_he_backend->decode(&val_decoded, *plaintext, type);
        EXPECT_EQ(val_decoded, val);
    }
} */
