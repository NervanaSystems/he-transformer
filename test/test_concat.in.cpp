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

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  Shape shape_a{2, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{2, 3};
  auto B = make_shared<op::Parameter>(element::f32, shape_b);
  Shape shape_c{2, 3};
  auto C = make_shared<op::Parameter>(element::f32, shape_c);
  Shape shape_r{2, 8};
  auto t = make_shared<op::Concat>(NodeVector{A, B, C}, 1);
  auto f = make_shared<Function>(t, ParameterVector{A, B, C});

  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({t}, {A, B, C}, backend.get(), true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto b = inputs[1];
    auto c = inputs[2];
    auto result = results[0];

    copy_data(a, vector<float>{2, 4, 8, 16});
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});

    backend->call(backend->compile(f), {result}, {a, b, c});
    EXPECT_TRUE(all_close(
        vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13},
        read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  Shape shape_a{2, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{3, 2};
  auto B = make_shared<op::Parameter>(element::f32, shape_b);
  Shape shape_c{3, 2};
  auto C = make_shared<op::Parameter>(element::f32, shape_c);
  Shape shape_r{8, 2};
  auto t = make_shared<op::Concat>(NodeVector{A, B, C}, 0);
  auto f = make_shared<Function>(t, ParameterVector{A, B, C});

  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({t}, {A, B, C}, backend.get(), true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto b = inputs[1];
    auto c = inputs[2];
    auto result = results[0];

    copy_data(a, vector<float>{2, 4, 8, 16});
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});

    backend->call(backend->compile(f), {result}, {a, b, c});
    EXPECT_TRUE(all_close(
        vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13},
        read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  Shape shape_a{4};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_b{6};
  auto B = make_shared<op::Parameter>(element::f32, shape_b);
  Shape shape_c{2};
  auto C = make_shared<op::Parameter>(element::f32, shape_c);
  Shape shape_r{12};
  auto t = make_shared<op::Concat>(NodeVector{A, B, C}, 0);
  auto f = make_shared<Function>(t, ParameterVector{A, B, C});

  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({t}, {A, B, C}, backend.get(), true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto b = inputs[1];
    auto c = inputs[2];
    auto result = results[0];

    copy_data(a, vector<float>{2, 4, 8, 16});
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    copy_data(c, vector<float>{18, 19});

    backend->call(backend->compile(f), {result}, {a, b, c});
    EXPECT_TRUE(
        all_close(vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19},
                  read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  Shape shape{1, 1, 1, 1};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto C = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Concat>(NodeVector{A, B, C}, 0);
  auto f = make_shared<Function>(t, ParameterVector{A, B, C});

  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({t}, {A, B, C}, backend.get(), true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto b = inputs[1];
    auto c = inputs[2];
    auto result = results[0];

    copy_data(a, vector<float>{1});
    copy_data(b, vector<float>{2});
    copy_data(c, vector<float>{3});

    backend->call(backend->compile(f), {result}, {a, b, c});
    EXPECT_TRUE(all_close(vector<float>{1, 2, 3}, read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  Shape shape{1, 1};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto C = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Concat>(NodeVector{A, B, C}, 0);
  auto f = make_shared<Function>(t, ParameterVector{A, B, C});

  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({t}, {A, B, C}, backend.get(), true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto b = inputs[1];
    auto c = inputs[2];
    auto result = results[0];

    copy_data(a, vector<float>{1});
    copy_data(b, vector<float>{2});
    copy_data(c, vector<float>{3});

    backend->call(backend->compile(f), {result}, {a, b, c});
    EXPECT_TRUE(all_close(vector<float>{1, 2, 3}, read_vector<float>(result)));
  }
}
