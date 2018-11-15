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

NGRAPH_TEST(${BACKEND_NAME}, sum_trivial) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{2, 2};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Sum>(a, AxisSet{});
  auto f = make_shared<Function>(t, op::ParameterVector{a});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a}, backend, true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{1, 2, 3, 4});
    backend->call(f, {t_result}, {t_a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(t_result));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_trivial_5d) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{2, 2, 2, 2, 2};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Sum>(a, AxisSet{});
  auto f = make_shared<Function>(t, op::ParameterVector{a});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a}, backend, true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_result = results[0];

    copy_data(t_a,
              vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    backend->call(f, {t_result}, {t_a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<float>(t_result));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_to_scalar) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{2, 2};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Sum>(a, AxisSet{0, 1});
  auto f = make_shared<Function>(t, op::ParameterVector{a});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a}, backend, true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{1, 2, 3, 4});
    backend->call(f, {t_result}, {t_a});
    EXPECT_EQ((vector<float>{10}), read_vector<float>(t_result));

    // For some reason I'm feeling extra paranoid about making sure reduction
    // doesn't clobber the input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(t_a));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_columns) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{3, 2};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Sum>(a, AxisSet{0});
  auto f = make_shared<Function>(t, op::ParameterVector{a});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a}, backend, true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{1, 2, 3, 4, 5, 6});
    backend->call(f, {t_result}, {t_a});
    EXPECT_EQ((vector<float>{9, 12}), read_vector<float>(t_result));

    // For some reason I'm feeling extra paranoid about making sure reduction
    // doesn't clobber the input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(t_a));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_rows) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{3, 2};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Sum>(a, AxisSet{1});
  auto f = make_shared<Function>(t, op::ParameterVector{a});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a}, backend, true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{1, 2, 3, 4, 5, 6});
    backend->call(f, {t_result}, {t_a});
    EXPECT_EQ((vector<float>{3, 7, 11}), read_vector<float>(t_result));

    // For some reason I'm feeling extra paranoid about making sure reduction
    // doesn't clobber the input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(t_a));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_rows_zero) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{3, 0};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Sum>(a, AxisSet{1});
  auto f = make_shared<Function>(t, op::ParameterVector{a});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a}, backend, true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{});
    copy_data(t_result, vector<float>({3, 3, 3}));
    backend->call(f, {t_result}, {t_a});
    EXPECT_TRUE(
        all_close((vector<float>{0, 0, 0}), read_vector<float>(t_result)));

    // For some reason I'm feeling extra paranoid about making sure reduction
    // doesn't clobber the input tensors, so let's do this too.
    EXPECT_TRUE(all_close((read_vector<float>(t_a)), vector<float>{}, 1e-5f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_cols_zero) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{0, 2};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Sum>(a, AxisSet{0});
  auto f = make_shared<Function>(t, op::ParameterVector{a});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a}, backend, true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{});
    copy_data(t_result, vector<float>({3, 3}));
    backend->call(f, {t_result}, {t_a});
    EXPECT_TRUE(all_close((vector<float>{0, 0}), read_vector<float>(t_result)));

    // For some reason I'm feeling extra paranoid about making sure reduction
    // doesn't clobber the input tensors, so let's do this too.
    EXPECT_TRUE(all_close((read_vector<float>(t_a)), vector<float>{}, 1e-5f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_vector_zero) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{0};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Sum>(a, AxisSet{0});
  auto f = make_shared<Function>(t, op::ParameterVector{a});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a}, backend, true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{});
    copy_data(t_result, vector<float>({3}));
    backend->call(f, {t_result}, {t_a});
    EXPECT_TRUE(all_close((vector<float>{0}), read_vector<float>(t_result)));

    // For some reason I'm feeling extra paranoid about making sure reduction
    // doesn't clobber the input tensors, so let's do this too.
    EXPECT_TRUE(all_close((read_vector<float>(t_a)), vector<float>{}, 1e-5f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_to_scalar_zero_by_zero) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{0, 0};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Sum>(a, AxisSet{0, 1});
  auto f = make_shared<Function>(t, op::ParameterVector{a});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a}, backend, true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{});
    copy_data(t_result, vector<float>({3}));
    backend->call(f, {t_result}, {t_a});
    EXPECT_TRUE(all_close((vector<float>{0}), read_vector<float>(t_result)));

    // For some reason I'm feeling extra paranoid about making sure reduction
    // doesn't clobber the input tensors, so let's do this too.
    EXPECT_TRUE(all_close((read_vector<float>(t_a)), vector<float>{}, 1e-5f));
  }
}