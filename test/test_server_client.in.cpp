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

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "he_op_annotations.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/op_annotations.hpp"
#include "op/bounded_relu.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_client.hpp"
#include "seal/he_seal_executable.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, server_client_add_3_multiple_parameters_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  size_t batch_size = 1;

  Shape shape{batch_size, 3};
  Shape shape_c{3, 1};
  auto a = op::Constant::create(element::f32, shape, {0.1, 0.2, 0.3});
  auto b = make_shared<op::Parameter>(element::f32, shape);

  auto from_client_annotation =
      std::make_shared<ngraph::he::HEOpAnnotations>(true);
  b->set_op_annotations(from_client_annotation);

  auto c = make_shared<op::Parameter>(element::f32, shape_c);
  auto d = make_shared<op::Reshape>(c, AxisVector{0, 1}, shape);
  auto t = make_shared<op::Add>(a, b);
  t = make_shared<op::Add>(t, d);
  auto f = make_shared<Function>(t, ParameterVector{b, c});

  auto t_c = he_backend->create_plain_tensor(element::f32, shape_c);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);
  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});
  copy_data(t_c, vector<float>{4, 5, 6});

  vector<float> results;
  auto client_thread = std::thread([&]() {
    vector<float> inputs{1, 2, 3};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {b->get_name(), std::make_pair("plain", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle =
      static_pointer_cast<ngraph::he::HESealExecutable>(he_backend->compile(f));
  handle->enable_client();
  handle->call_with_validate({t_result}, {t_dummy, t_c});
  client_thread.join();
  EXPECT_TRUE(all_close(results, vector<float>{5.1, 7.2, 9.3}, 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_add_3_multiple_parameters_encrypt) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  size_t batch_size = 1;

  Shape shape{batch_size, 3};
  Shape shape_c{3, 1};
  auto a = op::Constant::create(element::f32, shape, {0.1, 0.2, 0.3});
  auto b = make_shared<op::Parameter>(element::f32, shape);

  auto from_client_annotation =
      std::make_shared<ngraph::he::HEOpAnnotations>(true);
  b->set_op_annotations(from_client_annotation);

  auto c = make_shared<op::Parameter>(element::f32, shape_c);
  auto d = make_shared<op::Reshape>(c, AxisVector{0, 1}, shape);
  auto t = make_shared<op::Add>(a, b);
  t = make_shared<op::Add>(t, d);
  auto f = make_shared<Function>(t, ParameterVector{b, c});

  auto t_c = he_backend->create_plain_tensor(element::f32, shape_c);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);
  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});
  copy_data(t_c, vector<float>{4, 5, 6});

  vector<float> results;
  auto client_thread = std::thread([&]() {
    vector<float> inputs{1, 2, 3};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {b->get_name(), std::make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle =
      static_pointer_cast<ngraph::he::HESealExecutable>(he_backend->compile(f));
  handle->enable_client();
  handle->call_with_validate({t_result}, {t_dummy, t_c});
  client_thread.join();
  EXPECT_TRUE(all_close(results, vector<float>{5.1, 7.2, 9.3}, 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_add_3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  Shape shape{batch_size, 3};
  auto a = op::Constant::create(element::f32, shape, {0.1, 0.2, 0.3});
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Add>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{b});

  auto from_client_annotation =
      std::make_shared<ngraph::he::HEOpAnnotations>(true);
  b->set_op_annotations(from_client_annotation);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});

  vector<float> results;
  auto client_thread = std::thread([&]() {
    vector<float> inputs{1, 2, 3};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {b->get_name(), std::make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle =
      static_pointer_cast<ngraph::he::HESealExecutable>(he_backend->compile(f));
  handle->enable_client();
  handle->call_with_validate({t_result}, {t_dummy});
  client_thread.join();
  EXPECT_TRUE(all_close(results, vector<float>{1.1, 2.2, 3.3}, 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_add_3_relu) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  Shape shape{batch_size, 3};
  auto a = op::Constant::create(element::f32, shape, {0.1, 0.2, 0.3});
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Add>(a, b);
  auto relu = make_shared<op::Relu>(t);
  auto f = make_shared<Function>(relu, ParameterVector{b});

  auto from_client_annotation =
      std::make_shared<ngraph::he::HEOpAnnotations>(true);
  b->set_op_annotations(from_client_annotation);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});

  vector<float> results;
  auto client_thread = std::thread([&]() {
    vector<float> inputs{-1, -0.2, 3};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {b->get_name(), std::make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle =
      static_pointer_cast<ngraph::he::HESealExecutable>(he_backend->compile(f));
  handle->enable_client();
  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(all_close(results, vector<float>{0, 0, 3.3}, 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_add_3_relu_double) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  Shape shape{batch_size, 3};
  auto a = op::Constant::create(element::f64, shape, {0.1, 0.2, 0.3});
  auto b = make_shared<op::Parameter>(element::f64, shape);
  auto t = make_shared<op::Add>(a, b);
  auto relu = make_shared<op::Relu>(t);
  auto f = make_shared<Function>(relu, ParameterVector{b});

  auto from_client_annotation =
      std::make_shared<ngraph::he::HEOpAnnotations>(true);
  b->set_op_annotations(from_client_annotation);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f64, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f64, shape);

  // Used for dummy server inputs
  double DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<double>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});

  vector<double> results;
  auto client_thread = std::thread([&]() {
    vector<double> inputs{-1, -0.2, 3};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<double>{
            {b->get_name(), std::make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<double>(double_results.begin(), double_results.end());
  });

  auto handle =
      static_pointer_cast<ngraph::he::HESealExecutable>(he_backend->compile(f));
  handle->enable_client();
  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(all_close(results, vector<double>{0, 0, 3.3}, 1e-3));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_add_3_relu_int64_t) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  Shape shape{batch_size, 3};
  auto a = op::Constant::create(element::i64, shape, {1, 2, 3});
  auto b = make_shared<op::Parameter>(element::i64, shape);
  auto t = make_shared<op::Add>(a, b);
  auto relu = make_shared<op::Relu>(t);
  auto f = make_shared<Function>(relu, ParameterVector{b});

  auto from_client_annotation =
      std::make_shared<ngraph::he::HEOpAnnotations>(true);
  b->set_op_annotations(from_client_annotation);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::i64, shape);
  auto t_result = he_backend->create_cipher_tensor(element::i64, shape);

  // Used for dummy server inputs
  double DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<double>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});

  vector<int64_t> results;
  auto client_thread = std::thread([&]() {
    vector<int64_t> inputs{-1, 0, 3};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<int64_t>{
            {b->get_name(), std::make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results.resize(double_results.size());
    for (size_t i = 0; i < results.size(); ++i) {
      results[i] = std::round(double_results[i]);
    }
  });

  auto handle =
      static_pointer_cast<ngraph::he::HESealExecutable>(he_backend->compile(f));
  handle->enable_client();
  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(all_close(results, vector<int64_t>{0, 2, 6}, 0L));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_pad_bounded_relu) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  Shape shape{batch_size, 3};
  Shape result_shape{batch_size, 5};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = op::Constant::create(element::f32, Shape{}, vector<float>({0}));
  CoordinateDiff padding_below{0, 1};
  CoordinateDiff padding_above{0, 1};
  auto c = make_shared<op::Pad>(a, b, padding_below, padding_above);
  auto relu = make_shared<op::BoundedRelu>(c, 6.0f);
  auto f = make_shared<Function>(relu, ParameterVector{a});

  auto from_client_annotation =
      std::make_shared<ngraph::he::HEOpAnnotations>(true);
  a->set_op_annotations(from_client_annotation);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, result_shape);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});

  vector<float> results;
  auto client_thread = std::thread([&]() {
    vector<float> inputs{-1, 3, 7};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {a->get_name(), std::make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle =
      static_pointer_cast<ngraph::he::HESealExecutable>(he_backend->compile(f));
  handle->enable_client();
  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(all_close(results, vector<float>{0, 0, 3, 6, 0}, 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_pad_relu) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  Shape shape{batch_size, 3};
  Shape result_shape{batch_size, 5};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = op::Constant::create(element::f32, Shape{}, vector<float>({0}));
  CoordinateDiff padding_below{0, 1};
  CoordinateDiff padding_above{0, 1};
  auto c = make_shared<op::Pad>(a, b, padding_below, padding_above);
  auto relu = make_shared<op::Relu>(c);
  auto f = make_shared<Function>(relu, ParameterVector{a});

  auto from_client_annotation =
      std::make_shared<ngraph::he::HEOpAnnotations>(true);
  a->set_op_annotations(from_client_annotation);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, result_shape);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});

  vector<float> results;
  auto client_thread = std::thread([&]() {
    vector<float> inputs{-1, -0.2, 3};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {a->get_name(), std::make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle =
      static_pointer_cast<ngraph::he::HESealExecutable>(he_backend->compile(f));
  handle->enable_client();
  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(all_close(results, vector<float>{0, 0, 0, 3, 0}, 1e-3f));
}

auto server_client_relu_packed_test = [](size_t element_count,
                                         size_t batch_size,
                                         bool complex_packing, bool bounded,
                                         double bound_value = 0.) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        ngraph::he::HESealEncryptionParameters::
            default_complex_packing_parms());
  }

  Shape shape{batch_size, element_count};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  std::shared_ptr<ngraph::Function> f;
  if (bounded) {
    auto bounded_relu_op = make_shared<op::BoundedRelu>(a, bound_value);
    f = make_shared<Function>(bounded_relu_op, ParameterVector{a});
  } else {
    auto relu_op = make_shared<op::Relu>(a);
    f = make_shared<Function>(relu_op, ParameterVector{a});
  }

  auto from_client_annotation =
      std::make_shared<ngraph::he::HEOpAnnotations>(true);
  a->set_op_annotations(from_client_annotation);

  auto relu = [](double d) { return d > 0 ? d : 0.; };
  auto bounded_relu = [bound_value](double d) {
    return d > bound_value ? bound_value : (d > 0) ? d : 0.;
  };

  // Server inputs which are not used
  auto t_dummy = he_backend->create_packed_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_packed_cipher_tensor(element::f32, shape);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>(shape_size(shape), DUMMY_FLOAT));

  vector<float> inputs(shape_size(shape));
  vector<float> exp_results(shape_size(shape));
  for (size_t i = 0; i < shape_size(shape); ++i) {
    inputs[i] = static_cast<int>(i) - static_cast<int>(shape_size(shape)) / 2;

    if (bounded) {
      exp_results[i] = bounded_relu(inputs[i]);
    } else {
      exp_results[i] = relu(inputs[i]);
    }
  }

  vector<float> results;
  auto client_thread = std::thread([&]() {
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {a->get_name(), std::make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle =
      static_pointer_cast<ngraph::he::HESealExecutable>(he_backend->compile(f));
  handle->enable_client();
  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(all_close(results, exp_results));
};

NGRAPH_TEST(${BACKEND_NAME}, server_client_relu_packed_1_complex) {
  server_client_relu_packed_test(10, 1, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_relu_packed_1) {
  server_client_relu_packed_test(10, 1, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_relu_packed_2_complex) {
  server_client_relu_packed_test(10, 2, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_relu_packed_2) {
  server_client_relu_packed_test(10, 2, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_relu_packed_3_complex) {
  server_client_relu_packed_test(10, 3, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_relu_packed_3) {
  server_client_relu_packed_test(10, 3, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_relu_packed_all_slots_complex) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  size_t slot_count = he_backend->get_ckks_encoder()->slot_count() * 2;
  server_client_relu_packed_test(10, slot_count, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_relu_packed_all_slots) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  size_t slot_count = he_backend->get_ckks_encoder()->slot_count();
  server_client_relu_packed_test(10, slot_count, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_bounded_relu_packed_1_complex) {
  server_client_relu_packed_test(10, 1, true, true, 1.0);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_bounded_relu_packed_1) {
  server_client_relu_packed_test(10, 1, false, true, 2.0);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_bounded_relu_packed_2_complex) {
  server_client_relu_packed_test(10, 2, true, true, 3.0);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_bounded_relu_packed_2) {
  server_client_relu_packed_test(10, 2, false, true, 4.0);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_bounded_relu_packed_3_complex) {
  server_client_relu_packed_test(10, 3, true, true, 5.0);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_bounded_relu_packed_3) {
  server_client_relu_packed_test(10, 3, false, true, 6.0);
}

NGRAPH_TEST(${BACKEND_NAME},
            server_client_bounded_relu_packed_all_slots_complex) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  size_t slot_count = he_backend->get_ckks_encoder()->slot_count() * 2;
  server_client_relu_packed_test(10, slot_count, true, true, 7.0);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_bounded_relu_packed_all_slots) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  size_t slot_count = he_backend->get_ckks_encoder()->slot_count();
  server_client_relu_packed_test(10, slot_count, false, true, 8.0);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_relu_packed_845) {
  server_client_relu_packed_test(845, 1, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_relu_packed_10000) {
  server_client_relu_packed_test(10000, 1, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_bounded_relu_packed_845) {
  server_client_relu_packed_test(845, 1, true, true, 6.0);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_bounded_relu_packed_10000) {
  server_client_relu_packed_test(10000, 1, true, true, 6.0);
}

/*
NGRAPH_TEST(${BACKEND_NAME},
            server_client_pad_max_pool_1d_1channel_1image_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  size_t batch_size = 1;

  Shape shape_a{1, 1, 12};
  Shape window_shape{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  auto B = op::Constant::create(element::f32, Shape{}, vector<float>({0}));
  CoordinateDiff padding_below{0, 0, 1};
  CoordinateDiff padding_above{0, 0, 1};
  auto C = make_shared<op::Pad>(A, B, padding_below, padding_above);
  Shape shape_r{1, 1, 12};
  auto f = make_shared<Function>(make_shared<op::MaxPool>(C, window_shape),
                                 ParameterVector{A});

  auto from_client_annotation =
      std::make_shared<ngraph::he::HEOpAnnotations>(true);
  A->set_op_annotations(from_client_annotation);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape_a);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape_r);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>(shape_size(shape_a), DUMMY_FLOAT));

  vector<float> results;
  auto client_thread = std::thread([&]() {
    vector<float> inputs{-1, -1, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        client_float_input_map{{A->get_name(), inputs}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle =
      static_pointer_cast<ngraph::he::HESealExecutable>(he_backend->compile(f));
  handle->enable_client();
  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(all_close(
      results,
      test::NDArray<float, 3>({{{0, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}})
          .get_vector(),
      1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_max_pool_1d_1channel_1image_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  Shape shape_a{1, 1, 14};
  Shape window_shape{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{1, 1, 12};
  auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape),
                                 ParameterVector{A});

  auto from_client_annotation =
      std::make_shared<ngraph::he::HEOpAnnotations>(true);
  A->set_op_annotations(from_client_annotation);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape_a);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape_r);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>(shape_size(shape_a), DUMMY_FLOAT));

  vector<float> results;
  auto client_thread = std::thread([&]() {
    vector<float> inputs{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0};
    auto he_client =
        ngraph::he::HESealClient("localhost", 34000, batch_size,
client_float_input_map{{A->get_name(), inputs}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle =
      static_pointer_cast<ngraph::he::HESealExecutable>(he_backend->compile(f));
  handle->enable_client();
  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(all_close(
      results,
      test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}})
          .get_vector(),
      1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME},
            server_client_max_pool_1d_1channel_2image_plain_batched) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 2;

  Shape shape_a{2, 1, 14};
  Shape window_shape{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 1, 12};
  auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape),
                                 ParameterVector{A});

  auto from_client_annotation =
      std::make_shared<ngraph::he::HEOpAnnotations>(true);
  A->set_op_annotations(from_client_annotation);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_packed_plain_tensor(element::f32, shape_a);
  auto t_result =
      he_backend->create_packed_cipher_tensor(element::f32, shape_r);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>(shape_size(shape_a), DUMMY_FLOAT));

  vector<float> results;
  auto client_thread = std::thread([&]() {
    vector<float> inputs =
        test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                                 {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
            .get_vector();
    auto he_client =
        ngraph::he::HESealClient("localhost", 34000, batch_size,
client_float_input_map{{A->get_name(), inputs}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle =
      static_pointer_cast<ngraph::he::HESealExecutable>(he_backend->compile(f));
  handle->enable_client();
  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(all_close(
      results,
      (test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}},
                                {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
           .get_vector()),
      1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME},
            server_client_max_pool_1d_2channel_2image_plain_batched) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 2;

  Shape shape_a{2, 2, 14};
  Shape window_shape{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 2, 12};
  auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape),
                                 ParameterVector{A});

  auto from_client_annotation =
      std::make_shared<ngraph::he::HEOpAnnotations>(true);
  A->set_op_annotations(from_client_annotation);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_packed_plain_tensor(element::f32, shape_a);
  auto t_result =
      he_backend->create_packed_cipher_tensor(element::f32, shape_r);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>(shape_size(shape_a), DUMMY_FLOAT));

  vector<float> results;
  auto client_thread = std::thread([&]() {
    vector<float> inputs =
      test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},
                               {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
          .get_vector();
    auto he_client =
        ngraph::he::HESealClient("localhost", 34000, batch_size,
client_float_input_map{{A->get_name(), inputs}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle =
      static_pointer_cast<ngraph::he::HESealExecutable>(he_backend->compile(f));
  handle->enable_client();
  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(all_close(
      results,
      (test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0},
                                 {0, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1}},

                                {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2},
                                 {2, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2}}})
           .get_vector()),
      1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME},
            server_client_max_pool_2d_2channel_2image_plain_batched) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 2;

  Shape shape_a{2, 2, 5, 5};
  Shape window_shape{2, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 2, 4, 3};
  auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape),
                                 ParameterVector{A});

  auto from_client_annotation =
      std::make_shared<ngraph::he::HEOpAnnotations>(true);
  A->set_op_annotations(from_client_annotation);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_packed_plain_tensor(element::f32, shape_a);
  auto t_result =
      he_backend->create_packed_cipher_tensor(element::f32, shape_r);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>(shape_size(shape_a), DUMMY_FLOAT));


  vector<float> results;
  auto client_thread = std::thread([&]() {
    vector<float> inputs =
      test::NDArray<float, 4>({{{{0, 1, 0, 2, 1},  // img 0 chan 0
                                 {0, 3, 2, 0, 0},
                                 {2, 0, 0, 0, 1},
                                 {2, 0, 1, 1, 2},
                                 {0, 2, 1, 0, 0}},

                                {{0, 0, 0, 2, 0},  // img 0 chan 1
                                 {0, 2, 3, 0, 1},
                                 {2, 0, 1, 0, 2},
                                 {3, 1, 0, 0, 0},
                                 {2, 0, 0, 0, 0}}},

                               {{{0, 2, 1, 1, 0},  // img 1 chan 0
                                 {0, 0, 2, 0, 1},
                                 {0, 0, 1, 2, 3},
                                 {2, 0, 0, 3, 0},
                                 {0, 0, 0, 0, 0}},

                                {{2, 1, 0, 0, 1},  // img 1 chan 1
                                 {0, 2, 0, 0, 0},
                                 {1, 1, 2, 0, 2},
                                 {1, 1, 1, 0, 1},
                                 {1, 0, 0, 0, 2}}}})
          .get_vector();

    auto he_client =
        ngraph::he::HESealClient("localhost", 34000, batch_size,
client_float_input_map{{A->get_name(), inputs}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle =
      static_pointer_cast<ngraph::he::HESealExecutable>(he_backend->compile(f));
  handle->enable_client();
  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(all_close(results,
                        (test::NDArray<float, 4>({{{{3, 3, 2},  // img 0 chan 0
                                                    {3, 3, 2},
                                                    {2, 1, 2},
                                                    {2, 2, 2}},

                                                   {{3, 3, 3},  // img 0 chan 1
                                                    {3, 3, 3},
                                                    {3, 1, 2},
                                                    {3, 1, 0}}},

                                                  {{{2, 2, 2},  // img 1 chan 0
                                                    {2, 2, 3},
                                                    {2, 3, 3},
                                                    {2, 3, 3}},

                                                   {{2, 2, 1},  // img 1 chan 1
                                                    {2, 2, 2},
                                                    {2, 2, 2},
                                                    {1, 1, 2}}}})
                             .get_vector()),
                        1e-3f));
}
*/