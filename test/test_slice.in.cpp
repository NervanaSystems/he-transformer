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

NGRAPH_TEST(${BACKEND_NAME}, slice_scalar) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape_a{};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{};
  auto r = make_shared<op::Slice>(A, Coordinate{}, Coordinate{});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, vector<float>{312});
    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{312}), read_vector<float>(result));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape_a{4, 4};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{3, 2};
  auto r = make_shared<op::Slice>(A, Coordinate{0, 1}, Coordinate{3, 3});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               15, 16});
    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{2, 3, 6, 7, 10, 11}), read_vector<float>(result));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_vector) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape_a{16};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{12};
  auto r = make_shared<op::Slice>(A, Coordinate{2}, Coordinate{14});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(
        a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}),
              read_vector<float>(result));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_strided) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape_a{4, 4};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 2};
  auto r = make_shared<op::Slice>(A, Coordinate{1, 0}, Coordinate{4, 4},
                                  Strides{2, 3});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(
        a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{4, 7, 12, 15}), read_vector<float>(result));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape_a{4, 4, 4};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 2, 2};
  auto r = make_shared<op::Slice>(A, Coordinate{1, 1, 1}, Coordinate{3, 3, 3});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a,
              vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                            39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{21, 22, 25, 26, 37, 38, 41, 42}),
              read_vector<float>(result));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d_strided) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape_a{4, 4, 4};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 2, 2};
  auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4},
                                  Strides{2, 2, 2});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a,
              vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                            27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64});
    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 3, 9, 11, 33, 35, 41, 43}),
              read_vector<float>(result));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d_strided_different_strides) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape_a{4, 4, 4};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 2, 2};
  auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4},
                                  Strides{2, 2, 3});
  auto f = make_shared<Function>(r, op::ParameterVector{A});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a,
              vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                            27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64});
    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 4, 9, 12, 33, 36, 41, 44}),
              read_vector<float>(result));
  }
}
