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

#include "he_op_annotations.hpp"
#include "ngraph/ngraph.hpp"
#include "seal/he_seal_backend.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, dot1d_plain_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  Shape shape{4};
  Shape shape_r{};
  {
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Dot>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto t_a = he_backend->create_plain_tensor(element::f32, shape);
    auto t_b = he_backend->create_plain_tensor(element::f32, shape);
    auto t_result = he_backend->create_plain_tensor(element::f32, shape_r);
    copy_data(t_a, vector<float>{2, 2, 3, 4});
    copy_data(t_b, vector<float>{5, 6, 7, 8});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(
        all_close(read_vector<float>(t_result), vector<float>{75}, 1e-2f));
  }
  {
    auto a = make_shared<op::Parameter>(element::f64, shape);
    auto b = make_shared<op::Parameter>(element::f64, shape);
    auto t = make_shared<op::Dot>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto t_a = he_backend->create_plain_tensor(element::f64, shape);
    auto t_b = he_backend->create_plain_tensor(element::f64, shape);
    auto t_result = he_backend->create_plain_tensor(element::f64, shape_r);
    copy_data(t_a, vector<double>{2, 2, 3, 4});
    copy_data(t_b, vector<double>{5, 6, 7, 8});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(
        all_close(read_vector<double>(t_result), vector<double>{75}, 1e-2));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape{4};
  {
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Dot>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a, b}, backend.get());
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_b = inputs[1];
      auto t_result = results[0];
      copy_data(t_a, vector<float>{2, 2, 3, 4});
      copy_data(t_b, vector<float>{5, 6, 7, 8});
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a, t_b});
      EXPECT_TRUE(
          all_close(read_vector<float>(t_result), vector<float>{75}, 1e-2f));
    }
  }
  {
    auto a = make_shared<op::Parameter>(element::f64, shape);
    auto b = make_shared<op::Parameter>(element::f64, shape);
    auto t = make_shared<op::Dot>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a, b}, backend.get());
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_b = inputs[1];
      auto t_result = results[0];
      copy_data(t_a, vector<double>{2, 2, 3, 4});
      copy_data(t_b, vector<double>{5, 6, 7, 8});
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a, t_b});
      EXPECT_TRUE(
          all_close(read_vector<double>(t_result), vector<double>{75}, 1e-2));
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  Shape shape{4};
  {
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Dot>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a, b}, backend.get());
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_b = inputs[1];
      auto t_result = results[0];
      copy_data(t_a, vector<float>{1, 2, 3, 4});
      copy_data(t_b, vector<float>{-1, 0, 1, 2});
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a, t_b});
      EXPECT_TRUE(
          all_close(read_vector<float>(t_result), vector<float>{10}, 1e-2f));
    }
  }
  {
    auto a = make_shared<op::Parameter>(element::f64, shape);
    auto b = make_shared<op::Parameter>(element::f64, shape);
    auto t = make_shared<op::Dot>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a, b}, backend.get());
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_b = inputs[1];
      auto t_result = results[0];
      copy_data(t_a, vector<double>{1, 2, 3, 4});
      copy_data(t_b, vector<double>{-1, 0, 1, 2});
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a, t_b});
      EXPECT_TRUE(
          all_close(read_vector<double>(t_result), vector<double>{10}, 1e-2));
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, dot_matrix_vector) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape_a{4, 4};
  Shape shape_b{4};
  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto b = make_shared<op::Parameter>(element::f32, shape_b);
    auto t = make_shared<op::Dot>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a, b}, backend.get());
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_b = inputs[1];
      auto t_result = results[0];
      copy_data(t_a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                   14, 15, 16});
      copy_data(t_b, vector<float>{17, 18, 19, 20});
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a, t_b});
      EXPECT_TRUE(all_close(read_vector<float>(t_result),
                            (vector<float>{190, 486, 782, 1078}), 1e-2f));
    }
  }
  {
    auto a = make_shared<op::Parameter>(element::f64, shape_a);
    auto b = make_shared<op::Parameter>(element::f64, shape_b);
    auto t = make_shared<op::Dot>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a, b}, backend.get());
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_b = inputs[1];
      auto t_result = results[0];
      copy_data(t_a, vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                    14, 15, 16});
      copy_data(t_b, vector<double>{17, 18, 19, 20});
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a, t_b});
      EXPECT_TRUE(all_close(read_vector<double>(t_result),
                            (vector<double>{190, 486, 782, 1078}), 1e-2));
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape{};
  {
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Dot>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a, b}, backend.get());
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_b = inputs[1];
      auto t_result = results[0];
      copy_data(t_a, vector<float>{8});
      copy_data(t_b, vector<float>{6});
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a, t_b});
      EXPECT_TRUE(
          all_close(read_vector<float>(t_result), (vector<float>{48}), 1e-2f));
    }
  }
  {
    auto a = make_shared<op::Parameter>(element::f64, shape);
    auto b = make_shared<op::Parameter>(element::f64, shape);
    auto t = make_shared<op::Dot>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a, b}, backend.get());
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_b = inputs[1];
      auto t_result = results[0];
      copy_data(t_a, vector<double>{8});
      copy_data(t_b, vector<double>{6});
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a, t_b});
      EXPECT_TRUE(
          all_close(read_vector<double>(t_result), (vector<double>{48}), 1e-2));
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_packed) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  auto packed_plaintext_annotation =
      std::make_shared<ngraph::he::HEOpAnnotations>(false, false, true);

  Shape shape_a{3, 1};
  Shape shape_b{1};
  Shape shape_r{3};
  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    a->set_op_annotations(packed_plaintext_annotation);
    auto b = op::Constant::create(element::f32, shape_b, {4});
    auto t = make_shared<op::Dot>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a});
    auto t_a = he_backend->create_packed_plain_tensor(element::f32, shape_a);
    auto t_result =
        he_backend->create_packed_plain_tensor(element::f32, shape_r);
    copy_data(t_a, vector<float>{1, 2, 3});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_TRUE(all_close((vector<float>{4, 8, 12}),
                          read_vector<float>(t_result), 1e-2f));
  }
  {
    auto a = make_shared<op::Parameter>(element::f64, shape_a);
    a->set_op_annotations(packed_plaintext_annotation);
    auto b = op::Constant::create(element::f64, shape_b, {4});
    auto t = make_shared<op::Dot>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a});
    auto t_a = he_backend->create_packed_plain_tensor(element::f64, shape_a);
    auto t_result =
        he_backend->create_packed_plain_tensor(element::f64, shape_r);
    copy_data(t_a, vector<double>{1, 2, 3});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_TRUE(all_close((vector<double>{4, 8, 12}),
                          read_vector<double>(t_result), 1e-2));
  }
}
