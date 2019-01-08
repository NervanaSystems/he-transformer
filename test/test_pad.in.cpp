//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
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

#include "he_backend.hpp"
#include "ngraph/ngraph.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, pad_interior_1d) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{6};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = make_shared<op::Parameter>(element::f32, shape_b);
  Shape shape_r{16};
  Shape padding_below{0};
  Shape padding_above{0};
  Shape padding_interior{2};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A, B});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
  auto b = he_backend->create_cipher_tensor(element::f32, shape_b);
  copy_data(b, vector<float>{2112});
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  backend->call(backend->compile(f), {result}, {a, b});
  EXPECT_TRUE(all_close(
      (test::NDArray<float, 1>({1, 2112, 2112, 2, 2112, 2112, 3, 2112, 2112, 4,
                                2112, 2112, 5, 2112, 2112, 6})
           .get_vector()),
      read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_1d) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{6};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = make_shared<op::Parameter>(element::f32, shape_b);
  Shape shape_r{15};
  Shape padding_below{4};
  Shape padding_above{5};
  Shape padding_interior{0};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A, B});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
  auto b = he_backend->create_cipher_tensor(element::f32, shape_b);
  copy_data(b, vector<float>{2112});
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  backend->call(backend->compile(f), {result}, {a, b});
  EXPECT_TRUE(
      all_close((test::NDArray<float, 1>({2112, 2112, 2112, 2112, 1, 2, 3, 4, 5,
                                          6, 2112, 2112, 2112, 2112, 2112})
                     .get_vector()),
                read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_interior_exterior_1d) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{6};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = make_shared<op::Parameter>(element::f32, shape_b);
  Shape shape_r{25};
  Shape padding_below{4};
  Shape padding_above{5};
  Shape padding_interior{2};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A, B});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
  auto b = he_backend->create_cipher_tensor(element::f32, shape_b);
  copy_data(b, vector<float>{2112});
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  he_backend->call(backend->compile(f), {result}, {a, b});
  EXPECT_TRUE(all_close((test::NDArray<float, 1>(
                             {2112, 2112, 2112, 2112, 1,    2112, 2112, 2, 2112,
                              2112, 3,    2112, 2112, 4,    2112, 2112, 5, 2112,
                              2112, 6,    2112, 2112, 2112, 2112, 2112})
                             .get_vector()),
                        read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_interior_exterior_2d) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{2, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = make_shared<op::Parameter>(element::f32, shape_b);
  Shape shape_r{7, 6};
  Shape padding_below{1, 0};
  Shape padding_above{2, 1};
  Shape padding_interior{2, 1};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A, B});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  copy_data(a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
  auto b = he_backend->create_cipher_tensor(element::f32, shape_b);
  copy_data(b, vector<float>{9});
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  backend->call(backend->compile(f), {result}, {a, b});
  EXPECT_TRUE(all_close((test::NDArray<float, 2>({{9, 9, 9, 9, 9, 9},
                                                  {1, 9, 2, 9, 3, 9},
                                                  {9, 9, 9, 9, 9, 9},
                                                  {9, 9, 9, 9, 9, 9},
                                                  {4, 9, 5, 9, 6, 9},
                                                  {9, 9, 9, 9, 9, 9},
                                                  {9, 9, 9, 9, 9, 9}})
                             .get_vector()),
                        read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_0x0) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{0, 0};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = make_shared<op::Parameter>(element::f32, shape_b);
  Shape shape_r{5, 5};
  Shape padding_below{2, 3};
  Shape padding_above{3, 2};
  Shape padding_interior{0, 0};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A, B});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  // copy_data(a, test::NDArray<float, 2>({{}}).get_vector(), backend);
  auto b = he_backend->create_cipher_tensor(element::f32, shape_b);
  copy_data(b, vector<float>{2112});
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  backend->call(backend->compile(f), {result}, {a, b});
  EXPECT_TRUE(
      all_close((test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                          {2112, 2112, 2112, 2112, 2112},
                                          {2112, 2112, 2112, 2112, 2112},
                                          {2112, 2112, 2112, 2112, 2112},
                                          {2112, 2112, 2112, 2112, 2112}})
                     .get_vector()),
                read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_0x3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{0, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = make_shared<op::Parameter>(element::f32, shape_b);
  Shape shape_r{5, 5};
  Shape padding_below{2, 1};
  Shape padding_above{3, 1};
  Shape padding_interior{0, 0};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A, B});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  // copy_data(a, test::NDArray<float, 2>({}).get_vector(), backend);
  auto b = he_backend->create_cipher_tensor(element::f32, shape_b);
  copy_data(b, vector<float>{2112});
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  backend->call(backend->compile(f), {result}, {a, b});
  EXPECT_TRUE(
      all_close((test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                          {2112, 2112, 2112, 2112, 2112},
                                          {2112, 2112, 2112, 2112, 2112},
                                          {2112, 2112, 2112, 2112, 2112},
                                          {2112, 2112, 2112, 2112, 2112}})
                     .get_vector()),
                read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_3x0) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{3, 0};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = make_shared<op::Parameter>(element::f32, shape_b);
  Shape shape_r{5, 5};
  Shape padding_below{1, 3};
  Shape padding_above{1, 2};
  Shape padding_interior{0, 0};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A, B});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  // copy_data(a, test::NDArray<float, 2>({}).get_vector(), backend);
  auto b = he_backend->create_cipher_tensor(element::f32, shape_b);
  copy_data(b, vector<float>{2112});
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  backend->call(backend->compile(f), {result}, {a, b});
  EXPECT_TRUE(
      all_close((test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                          {2112, 2112, 2112, 2112, 2112},
                                          {2112, 2112, 2112, 2112, 2112},
                                          {2112, 2112, 2112, 2112, 2112},
                                          {2112, 2112, 2112, 2112, 2112}})
                     .get_vector()),
                read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_4d_1x2x2x2) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{1, 2, 2, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = make_shared<op::Parameter>(element::f32, shape_b);
  Shape shape_r{1, 2, 4, 4};
  Shape padding_below{0, 0, 1, 1};
  Shape padding_above{0, 0, 1, 1};
  Shape padding_interior{0, 0, 0, 0};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A, B});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  // clang-format off
    copy_data(a, test::NDArray<float, 4>(
        {
            {
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                },
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                }
            }
        }).get_vector());
  // clang-format on

  auto b = he_backend->create_cipher_tensor(element::f32, shape_b);
  copy_data(b, vector<float>{42});

  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  backend->call(backend->compile(f), {result}, {a, b});
  // clang-format off
    EXPECT_TRUE(all_close((test::NDArray<float, 4>(
        {
            {
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                },
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                }
            }
        }).get_vector()),
        read_vector<float>(result), 1e-3f));
  // clang-format on
}

// This is a regression test for one of TF's unit tests, which was failing.
// The problem was inappropriate handling of the shape computation for a
// zero-length axis with interior padding. Rather than subtract 1 from the
// source shape and multiply by the interior padding (which causes underflow),
// we should just count the pre-interior-padding length as zero.
NGRAPH_TEST(${BACKEND_NAME}, pad_interior_exterior_4d_2x0x3x2) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{2, 0, 3, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = make_shared<op::Parameter>(element::f32, shape_b);
  Shape padding_below{1, 0, 0, 0};
  Shape padding_above{0, 2, 0, 0};
  Shape padding_interior{2, 1, 0, 0};
  Shape shape_r{5, 2, 3, 2};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A, B});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  // copy_data(a, test::NDArray<float, 2>({}).get_vector(), backend);
  auto b = he_backend->create_cipher_tensor(element::f32, shape_b);
  copy_data(b, vector<float>{2112});
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  vector<float> expected(5 * 2 * 3 * 2, 2112);

  backend->call(backend->compile(f), {result}, {a, b});
  EXPECT_TRUE(all_close(expected, read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_const_interior_1d) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{6};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = op::Constant::create(element::f32, shape_b, {123});
  Shape shape_r{16};
  Shape padding_below{0};
  Shape padding_above{0};
  Shape padding_interior{2};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  backend->call(backend->compile(f), {result}, {a});
  EXPECT_TRUE(
      all_close((test::NDArray<float, 1>({1, 123, 123, 2, 123, 123, 3, 123, 123,
                                          4, 123, 123, 5, 123, 123, 6})
                     .get_vector()),
                read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_const_exterior_1d) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{6};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = op::Constant::create(element::f32, shape_b, {123});
  Shape shape_r{15};
  Shape padding_below{4};
  Shape padding_above{5};
  Shape padding_interior{0};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  backend->call(backend->compile(f), {result}, {a});
  EXPECT_TRUE(
      all_close((test::NDArray<float, 1>({123, 123, 123, 123, 1, 2, 3, 4, 5, 6,
                                          123, 123, 123, 123, 123})
                     .get_vector()),
                read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_const_interior_exterior_1d) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{6};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = op::Constant::create(element::f32, shape_b, {123});
  Shape shape_r{25};
  Shape padding_below{4};
  Shape padding_above{5};
  Shape padding_interior{2};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  backend->call(backend->compile(f), {result}, {a});
  EXPECT_TRUE(all_close(
      (test::NDArray<float, 1>({123, 123, 123, 123, 1,   123, 123, 2, 123,
                                123, 3,   123, 123, 4,   123, 123, 5, 123,
                                123, 6,   123, 123, 123, 123, 123})
           .get_vector()),
      read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_const_interior_exterior_2d) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{2, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = op::Constant::create(element::f32, shape_b, {123});
  Shape shape_r{7, 6};
  Shape padding_below{1, 0};
  Shape padding_above{2, 1};
  Shape padding_interior{2, 1};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  copy_data(a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  backend->call(backend->compile(f), {result}, {a});
  EXPECT_TRUE(
      all_close((test::NDArray<float, 2>({{123, 123, 123, 123, 123, 123},
                                          {1, 123, 2, 123, 3, 123},
                                          {123, 123, 123, 123, 123, 123},
                                          {123, 123, 123, 123, 123, 123},
                                          {4, 123, 5, 123, 6, 123},
                                          {123, 123, 123, 123, 123, 123},
                                          {123, 123, 123, 123, 123, 123}})
                     .get_vector()),
                read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_const_exterior_2d_0x0) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{0, 0};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = op::Constant::create(element::f32, shape_b, {123});
  Shape shape_r{5, 5};
  Shape padding_below{2, 3};
  Shape padding_above{3, 2};
  Shape padding_interior{0, 0};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  // copy_data(a, test::NDArray<float, 2>({{}}).get_vector(), backend);
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  backend->call(backend->compile(f), {result}, {a});
  EXPECT_TRUE(all_close((test::NDArray<float, 2>({{123, 123, 123, 123, 123},
                                                  {123, 123, 123, 123, 123},
                                                  {123, 123, 123, 123, 123},
                                                  {123, 123, 123, 123, 123},
                                                  {123, 123, 123, 123, 123}})
                             .get_vector()),
                        read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_const_exterior_2d_0x3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{0, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = op::Constant::create(element::f32, shape_b, {123});
  Shape shape_r{5, 5};
  Shape padding_below{2, 1};
  Shape padding_above{3, 1};
  Shape padding_interior{0, 0};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  // copy_data(a, test::NDArray<float, 2>({}).get_vector(), backend);
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  backend->call(backend->compile(f), {result}, {a});
  EXPECT_TRUE(all_close((test::NDArray<float, 2>({{123, 123, 123, 123, 123},
                                                  {123, 123, 123, 123, 123},
                                                  {123, 123, 123, 123, 123},
                                                  {123, 123, 123, 123, 123},
                                                  {123, 123, 123, 123, 123}})
                             .get_vector()),
                        read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_const_exterior_2d_3x0) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{3, 0};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = op::Constant::create(element::f32, shape_b, {123});
  Shape shape_r{5, 5};
  Shape padding_below{1, 3};
  Shape padding_above{1, 2};
  Shape padding_interior{0, 0};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  // copy_data(a, test::NDArray<float, 2>({}).get_vector(), backend);
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  backend->call(backend->compile(f), {result}, {a});
  EXPECT_TRUE(all_close((test::NDArray<float, 2>({{123, 123, 123, 123, 123},
                                                  {123, 123, 123, 123, 123},
                                                  {123, 123, 123, 123, 123},
                                                  {123, 123, 123, 123, 123},
                                                  {123, 123, 123, 123, 123}})
                             .get_vector()),
                        read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_const_exterior_4d_1x2x2x2) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{1, 2, 2, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = op::Constant::create(element::f32, shape_b, {123});
  Shape shape_r{1, 2, 4, 4};
  Shape padding_below{0, 0, 1, 1};
  Shape padding_above{0, 0, 1, 1};
  Shape padding_interior{0, 0, 0, 0};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  // clang-format off
    copy_data(a, test::NDArray<float, 4>(
        {
            {
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                },
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                }
            }
        }).get_vector());
  // clang-format on

  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  backend->call(backend->compile(f), {result}, {a});
  // clang-format off
    EXPECT_TRUE(all_close((test::NDArray<float, 4>(
        {
            {
                {
                    {123.0f, 123.0f, 123.0f, 123.0f},
                    {123.0f, 0.0f, 0.0f, 123.0f},
                    {123.0f, 0.0f, 0.0f, 123.0f},
                    {123.0f, 123.0f, 123.0f, 123.0f}
                },
                {
                    {123.0f, 123.0f, 123.0f, 123.0f},
                    {123.0f, 0.0f, 0.0f, 123.0f},
                    {123.0f, 0.0f, 0.0f, 123.0f},
                    {123.0f, 123.0f, 123.0f, 123.0f}
                }
            }
        }).get_vector()),
        read_vector<float>(result), 1e-3f));
  // clang-format on
}

// This is a regression test for one of TF's unit tests, which was failing.
// The problem was inappropriate handling of the shape computation for a
// zero-length axis with interior padding. Rather than subtract 1 from the
// source shape and multiply by the interior padding (which causes underflow),
// we should just count the pre-interior-padding length as zero.
NGRAPH_TEST(${BACKEND_NAME}, pad_const_interior_exterior_4d_2x0x3x2) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{2, 0, 3, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{};
  auto B = op::Constant::create(element::f32, shape_b, {123});
  Shape padding_below{1, 0, 0, 0};
  Shape padding_above{0, 2, 0, 0};
  Shape padding_interior{2, 1, 0, 0};
  Shape shape_r{5, 2, 3, 2};
  auto f = make_shared<Function>(
      make_shared<op::Pad>(A, B, padding_below, padding_above,
                           padding_interior),
      ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  // copy_data(a, test::NDArray<float, 2>({}).get_vector(), backend);
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  vector<float> expected(5 * 2 * 3 * 2, 123);

  backend->call(backend->compile(f), {result}, {a});
  EXPECT_TRUE(all_close(expected, read_vector<float>(result), 1e-3f));
}
