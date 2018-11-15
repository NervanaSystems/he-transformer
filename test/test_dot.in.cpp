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

#include "seal/ckks/he_seal_ckks_backend.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, dot1d) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{4};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Dot>(a, b);
  auto f = make_shared<Function>(t, op::ParameterVector{a, b});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_b = inputs[1];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{1, 2, 3, 4});
    copy_data(t_b, vector<float>{5, 6, 7, 8});
    backend->call(f, {t_result}, {t_a, t_b});
    EXPECT_TRUE(
        all_close(read_vector<float>(t_result), vector<float>{70}, 1e-4f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{4};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Dot>(a, b);
  auto f = make_shared<Function>(t, op::ParameterVector{a, b});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_b = inputs[1];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{1, 2, 3, 4});
    copy_data(t_b, vector<float>{-1, 0, 1, 2});
    backend->call(f, {t_result}, {t_a, t_b});
    EXPECT_TRUE(
        all_close(read_vector<float>(t_result), vector<float>{10}, 0.001f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, dot_matrix_vector) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape_a{4, 4};
  Shape shape_b{4};

  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto b = make_shared<op::Parameter>(element::f32, shape_b);
  auto t = make_shared<op::Dot>(a, b);
  auto f = make_shared<Function>(t, op::ParameterVector{a, b});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_b = inputs[1];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                 15, 16});
    copy_data(t_b, vector<float>{17, 18, 19, 20});
    backend->call(f, {t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close(read_vector<float>(t_result),
                          (vector<float>{190, 486, 782, 1078}), 1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{};

  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Dot>(a, b);
  auto f = make_shared<Function>(t, op::ParameterVector{a, b});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_b = inputs[1];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{8});
    copy_data(t_b, vector<float>{6});
    backend->call(f, {t_result}, {t_a, t_b});
    EXPECT_TRUE(
        all_close(read_vector<float>(t_result), (vector<float>{48}), 1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_batch) {
  auto backend = static_pointer_cast<runtime::he::he_seal::HESealCKKSBackend>(
      runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

  Shape shape_a{3, 1};
  Shape shape_b{1};
  Shape shape_r{3, 1};
  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto b = make_shared<op::Parameter>(element::f32, shape_b);
  auto t = make_shared<op::Dot>(a, b);

  auto f = make_shared<Function>(t, op::ParameterVector{a, b});

  // Create some tensors for input/output
  auto t_a = backend->create_batched_tensor(element::f32, shape_a);
  auto t_b = backend->create_plain_tensor(element::f32, shape_b);
  auto t_result = backend->create_batched_tensor(element::f32, shape_r);

  copy_data(t_a, vector<float>{1, 2, 3});
  copy_data(t_b, vector<float>{4});
  backend->call(f, {t_result}, {t_a, t_b});
  EXPECT_TRUE(all_close((vector<float>{4, 8, 12}),
                        generalized_read_vector<float>(t_result), 1e-3f));
}