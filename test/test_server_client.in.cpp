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
#include <cmath>
#include <memory>
#include <thread>
#include <vector>

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

static std::string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, server_client_bad_configuration) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  ngraph::Shape shape{3, 1};
  auto b = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);

  std::string error_str;
  EXPECT_ANY_THROW(
      he_backend->set_config({{"enable_client", "false"},
                              {b->get_name(), "client_input,plain,packed"}},
                             error_str));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_duplicate_setup) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  ngraph::Shape shape{batch_size, 3};
  ngraph::Shape shape_c{3, 1};
  auto a = ngraph::op::Constant::create(ngraph::element::f32, shape,
                                        {0.1, 0.2, 0.3});
  auto b = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);

  std::string error_str;
  he_backend->set_config(
      {{"enable_client", "true"}, {b->get_name(), "client_input,plain,packed"}},
      error_str);

  auto c =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape_c);
  auto d =
      std::make_shared<ngraph::op::Reshape>(c, ngraph::AxisVector{0, 1}, shape);
  auto t = std::make_shared<ngraph::op::Add>(a, b);
  t = std::make_shared<ngraph::op::Add>(t, d);
  auto f = std::make_shared<ngraph::Function>(t, ngraph::ParameterVector{b, c});

  auto t_c = he_backend->create_plain_tensor(ngraph::element::f32, shape_c);
  auto t_result = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(ngraph::element::f32, shape);

  // Used for dummy server inputs
  float dummy_float = 99;
  copy_data(t_dummy, std::vector<float>{dummy_float, dummy_float, dummy_float});
  copy_data(t_c, std::vector<float>{4, 5, 6});

  std::vector<float> results;
  auto client_thread = std::thread([&]() {
    std::vector<float> inputs{1, 2, 3};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {b->get_name(), make_pair("plain", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = std::static_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  // Set up server again
  EXPECT_TRUE(handle->server_setup());

  handle->call_with_validate({t_result}, {t_dummy, t_c});
  client_thread.join();
  EXPECT_TRUE(ngraph::test::he::all_close(
      results, std::vector<float>{5.1, 7.2, 9.3}, 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_provenance_tag) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  ngraph::Shape shape{batch_size, 3};
  ngraph::Shape shape_c{3, 1};
  auto a = ngraph::op::Constant::create(ngraph::element::f32, shape,
                                        {0.1, 0.2, 0.3});
  auto b = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);

  std::string b_provenance_tag{"b_provenance_tag"};
  b->add_provenance_tag(b_provenance_tag);

  std::string error_str;
  he_backend->set_config({{"enable_client", "true"},
                          {b_provenance_tag, "client_input,plain,packed"}},
                         error_str);

  auto c =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape_c);
  auto d =
      std::make_shared<ngraph::op::Reshape>(c, ngraph::AxisVector{0, 1}, shape);
  auto t = std::make_shared<ngraph::op::Add>(a, b);
  t = std::make_shared<ngraph::op::Add>(t, d);
  auto f = std::make_shared<ngraph::Function>(t, ngraph::ParameterVector{b, c});

  auto t_c = he_backend->create_plain_tensor(ngraph::element::f32, shape_c);
  auto t_result = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(ngraph::element::f32, shape);

  // Used for dummy server inputs
  float dummy_float = 99;
  copy_data(t_dummy, std::vector<float>{dummy_float, dummy_float, dummy_float});
  copy_data(t_c, std::vector<float>{4, 5, 6});

  std::vector<float> results;
  auto client_thread = std::thread([&]() {
    std::vector<float> inputs{1, 2, 3};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {b_provenance_tag, make_pair("plain", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = std::static_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy, t_c});
  client_thread.join();
  EXPECT_TRUE(ngraph::test::he::all_close(
      results, std::vector<float>{5.1, 7.2, 9.3}, 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_add_3_multiple_parameters_plain) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  std::string error_str;

  size_t batch_size = 1;

  ngraph::Shape shape{batch_size, 3};
  ngraph::Shape shape_c{3, 1};
  auto a = ngraph::op::Constant::create(ngraph::element::f32, shape,
                                        {0.1, 0.2, 0.3});
  auto b = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto c =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape_c);
  auto d =
      std::make_shared<ngraph::op::Reshape>(c, ngraph::AxisVector{0, 1}, shape);
  auto t = std::make_shared<ngraph::op::Add>(a, b);
  t = std::make_shared<ngraph::op::Add>(t, d);
  auto f = std::make_shared<ngraph::Function>(t, ngraph::ParameterVector{b, c});

  he_backend->set_config(
      {{"enable_client", "true"}, {b->get_name(), "client_input,plain"}},
      error_str);

  auto t_c = he_backend->create_plain_tensor(ngraph::element::f32, shape_c);
  auto t_result = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(ngraph::element::f32, shape);

  // Used for dummy server inputs
  float dummy_float = 99;
  copy_data(t_dummy, std::vector<float>{dummy_float, dummy_float, dummy_float});
  copy_data(t_c, std::vector<float>{4, 5, 6});

  std::vector<float> results;
  auto client_thread = std::thread([&]() {
    std::vector<float> inputs{1, 2, 3};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {b->get_name(), make_pair("plain", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = std::static_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy, t_c});
  client_thread.join();
  EXPECT_TRUE(ngraph::test::he::all_close(
      results, std::vector<float>{5.1, 7.2, 9.3}, 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_add_3_multiple_parameters_encrypt) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  std::string config_str = R"(
    {
        "scheme_name" : "HE_SEAL",
        "poly_modulus_degree" : 2048,
        "security_level" : 0,
        "coeff_modulus" : [30, 30, 30, 30, 30],
        "scale" : 1073741824
    })";

  size_t batch_size = 1;

  ngraph::Shape shape{batch_size, 3};
  ngraph::Shape shape_c{3, 1};
  auto a = ngraph::op::Constant::create(ngraph::element::f32, shape,
                                        {0.1, 0.2, 0.3});
  auto b = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto c =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape_c);
  auto d =
      std::make_shared<ngraph::op::Reshape>(c, ngraph::AxisVector{0, 1}, shape);
  auto t = std::make_shared<ngraph::op::Add>(a, b);
  t = std::make_shared<ngraph::op::Add>(t, d);
  auto f = std::make_shared<ngraph::Function>(t, ngraph::ParameterVector{b, c});

  std::string error_str;
  he_backend->set_config({{"enable_client", "true"},
                          {"encryption_parameters", config_str},
                          {b->get_name(), "client_input,encrypt"}},
                         error_str);

  auto t_c = he_backend->create_plain_tensor(ngraph::element::f32, shape_c);
  auto t_result = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(ngraph::element::f32, shape);

  // Used for dummy server inputs
  float dummy_float = 99;
  copy_data(t_dummy, std::vector<float>{dummy_float, dummy_float, dummy_float});
  copy_data(t_c, std::vector<float>{4, 5, 6});

  std::vector<float> results;
  auto client_thread = std::thread([&]() {
    std::vector<float> inputs{1, 2, 3};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {b->get_name(), make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = std::static_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  // Inject extra server setup setep
  handle->server_setup();
  handle->call_with_validate({t_result}, {t_dummy, t_c});
  client_thread.join();
  EXPECT_TRUE(ngraph::test::he::all_close(
      results, std::vector<float>{5.1, 7.2, 9.3}, 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_add_3) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  ngraph::Shape shape{batch_size, 3};
  auto a = ngraph::op::Constant::create(ngraph::element::f32, shape,
                                        {0.1, 0.2, 0.3});
  auto b = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto t = std::make_shared<ngraph::op::Add>(a, b);
  auto f = std::make_shared<ngraph::Function>(t, ngraph::ParameterVector{b});

  std::string error_str;
  he_backend->set_config(
      {{"enable_client", "true"}, {b->get_name(), "client_input,encrypt"}},
      error_str);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(ngraph::element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(ngraph::element::f32, shape);

  // Used for dummy server inputs
  float dummy_float = 99;
  copy_data(t_dummy, std::vector<float>{dummy_float, dummy_float, dummy_float});

  std::vector<float> results;
  auto client_thread = std::thread([&]() {
    std::vector<float> inputs{1, 2, 3};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {b->get_name(), make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = std::static_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy});
  client_thread.join();
  EXPECT_TRUE(ngraph::test::he::all_close(
      results, std::vector<float>{1.1, 2.2, 3.3}, 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_add_3_relu) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  ngraph::Shape shape{batch_size, 3};
  auto a = ngraph::op::Constant::create(ngraph::element::f32, shape,
                                        {0.1, 0.2, 0.3});
  auto b = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto t = std::make_shared<ngraph::op::Add>(a, b);
  auto relu = std::make_shared<ngraph::op::Relu>(t);
  auto f = std::make_shared<ngraph::Function>(relu, ngraph::ParameterVector{b});

  std::string error_str;
  he_backend->set_config(
      {{"enable_client", "true"}, {b->get_name(), "client_input,encrypt"}},
      error_str);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(ngraph::element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(ngraph::element::f32, shape);

  // Used for dummy server inputs
  float dummy_float = 99;
  copy_data(t_dummy, std::vector<float>{dummy_float, dummy_float, dummy_float});

  std::vector<float> results;
  auto client_thread = std::thread([&]() {
    std::vector<float> inputs{-1, -0.2, 3};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {b->get_name(), make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = std::static_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(ngraph::test::he::all_close(
      results, std::vector<float>{0, 0, 3.3}, 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_add_3_relu_double) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  ngraph::Shape shape{batch_size, 3};
  auto a = ngraph::op::Constant::create(ngraph::element::f64, shape,
                                        {0.1, 0.2, 0.3});
  auto b = std::make_shared<ngraph::op::Parameter>(ngraph::element::f64, shape);
  auto t = std::make_shared<ngraph::op::Add>(a, b);
  auto relu = std::make_shared<ngraph::op::Relu>(t);
  auto f = std::make_shared<ngraph::Function>(relu, ngraph::ParameterVector{b});

  std::string error_str;
  he_backend->set_config(
      {{"enable_client", "true"}, {b->get_name(), "client_input,encrypt"}},
      error_str);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(ngraph::element::f64, shape);
  auto t_result = he_backend->create_cipher_tensor(ngraph::element::f64, shape);

  // Used for dummy server inputs
  double dummy_float = 99;
  copy_data(t_dummy,
            std::vector<double>{dummy_float, dummy_float, dummy_float});

  std::vector<double> results;
  auto client_thread = std::thread([&]() {
    std::vector<double> inputs{-1, -0.2, 3};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<double>{
            {b->get_name(), make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<double>(double_results.begin(), double_results.end());
  });

  auto handle = std::static_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(ngraph::test::he::all_close(
      results, std::vector<double>{0, 0, 3.3}, 1e-3));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_add_3_relu_int64_t) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  ngraph::Shape shape{batch_size, 3};
  auto a = ngraph::op::Constant::create(ngraph::element::i64, shape, {1, 2, 3});
  auto b = std::make_shared<ngraph::op::Parameter>(ngraph::element::i64, shape);
  auto t = std::make_shared<ngraph::op::Add>(a, b);
  auto relu = std::make_shared<ngraph::op::Relu>(t);
  auto f = std::make_shared<ngraph::Function>(relu, ngraph::ParameterVector{b});

  std::string error_str;
  he_backend->set_config(
      {{"enable_client", "true"}, {b->get_name(), "client_input,encrypt"}},
      error_str);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(ngraph::element::i64, shape);
  auto t_result = he_backend->create_cipher_tensor(ngraph::element::i64, shape);

  // Used for dummy server inputs
  double dummy_float = 99;
  copy_data(t_dummy,
            std::vector<double>{dummy_float, dummy_float, dummy_float});

  std::vector<int64_t> results;
  auto client_thread = std::thread([&]() {
    std::vector<int64_t> inputs{-1, 0, 3};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<int64_t>{
            {b->get_name(), make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results.resize(double_results.size());
    for (size_t i = 0; i < results.size(); ++i) {
      results[i] = round(double_results[i]);
    }
  });

  auto handle = std::static_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(
      ngraph::test::he::all_close(results, std::vector<int64_t>{0, 2, 6}, 0L));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_pad_bounded_relu) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  ngraph::Shape shape{batch_size, 3};
  ngraph::Shape result_shape{batch_size, 5};
  auto a = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto b = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{},
                                        std::vector<float>({0}));
  ngraph::CoordinateDiff padding_below{0, 1};
  ngraph::CoordinateDiff padding_above{0, 1};
  auto c =
      std::make_shared<ngraph::op::Pad>(a, b, padding_below, padding_above);
  auto relu = std::make_shared<ngraph::op::BoundedRelu>(c, 6.0f);
  auto f = std::make_shared<ngraph::Function>(relu, ngraph::ParameterVector{a});

  std::string error_str;
  he_backend->set_config(
      {{"enable_client", "true"}, {a->get_name(), "client_input,encrypt"}},
      error_str);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(ngraph::element::f32, shape);
  auto t_result =
      he_backend->create_cipher_tensor(ngraph::element::f32, result_shape);

  // Used for dummy server inputs
  float dummy_float = 99;
  copy_data(t_dummy, std::vector<float>{dummy_float, dummy_float, dummy_float});

  std::vector<float> results;
  auto client_thread = std::thread([&]() {
    std::vector<float> inputs{-1, 3, 7};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {a->get_name(), make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = std::static_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(ngraph::test::he::all_close(
      results, std::vector<float>{0, 0, 3, 6, 0}, 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_pad_relu) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  ngraph::Shape shape{batch_size, 3};
  ngraph::Shape result_shape{batch_size, 5};
  auto a = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto b = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{},
                                        std::vector<float>({0}));
  ngraph::CoordinateDiff padding_below{0, 1};
  ngraph::CoordinateDiff padding_above{0, 1};
  auto c =
      std::make_shared<ngraph::op::Pad>(a, b, padding_below, padding_above);
  auto relu = std::make_shared<ngraph::op::Relu>(c);
  auto f = std::make_shared<ngraph::Function>(relu, ngraph::ParameterVector{a});

  std::string error_str;
  he_backend->set_config(
      {{"enable_client", "true"}, {a->get_name(), "client_input,encrypt"}},
      error_str);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(ngraph::element::f32, shape);
  auto t_result =
      he_backend->create_cipher_tensor(ngraph::element::f32, result_shape);

  // Used for dummy server inputs
  float dummy_float = 99;
  copy_data(t_dummy, std::vector<float>{dummy_float, dummy_float, dummy_float});

  std::vector<float> results;
  auto client_thread = std::thread([&]() {
    std::vector<float> inputs{-1, -0.2, 3};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {a->get_name(), make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = std::static_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(ngraph::test::he::all_close(
      results, std::vector<float>{0, 0, 0, 3, 0}, 1e-3f));
}

auto server_client_relu_packed_test = [](size_t element_count,
                                         size_t batch_size,
                                         bool complex_packing, bool bounded,
                                         double bound_value = 0.) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        ngraph::he::HESealEncryptionParameters::
            default_complex_packing_parms());
  }

  ngraph::Shape shape{batch_size, element_count};
  auto a = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  std::shared_ptr<ngraph::Function> f;
  if (bounded) {
    auto bounded_relu_op =
        std::make_shared<ngraph::op::BoundedRelu>(a, bound_value);
    f = std::make_shared<ngraph::Function>(bounded_relu_op,
                                           ngraph::ParameterVector{a});
  } else {
    auto relu_op = std::make_shared<ngraph::op::Relu>(a);
    f = std::make_shared<ngraph::Function>(relu_op, ngraph::ParameterVector{a});
  }

  bool packed = batch_size > 1;

  std::string tensor_config{"client_input,encrypt"};
  if (packed) {
    tensor_config.append(",packed");
  }
  std::string error_str;
  he_backend->set_config(
      {{"enable_client", "true"}, {a->get_name(), tensor_config}}, error_str);

  auto relu = [](double d) { return d > 0 ? d : 0.; };
  auto bounded_relu = [bound_value](double d) {
    return d > bound_value ? bound_value : (d > 0) ? d : 0.;
  };

  // Server inputs which are not used
  auto t_dummy =
      he_backend->create_packed_plain_tensor(ngraph::element::f32, shape);
  auto t_result =
      he_backend->create_packed_cipher_tensor(ngraph::element::f32, shape);

  // Used for dummy server inputs
  float dummy_float = 99;
  copy_data(t_dummy, std::vector<float>(shape_size(shape), dummy_float));

  std::vector<float> inputs(shape_size(shape));
  std::vector<float> exp_results(shape_size(shape));
  for (size_t i = 0; i < shape_size(shape); ++i) {
    inputs[i] =
        static_cast<float>(i) - static_cast<float>(shape_size(shape)) / 2.0;
    if (inputs[i] > 1000) {
      inputs[i] = fmod(inputs[i], 1000);
    }

    if (bounded) {
      exp_results[i] = bounded_relu(inputs[i]);
    } else {
      exp_results[i] = relu(inputs[i]);
    }
  }

  std::vector<float> results;
  auto client_thread = std::thread([&]() {
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {a->get_name(), make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = std::static_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(ngraph::test::he::all_close(results, exp_results));
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
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  size_t slot_count = he_backend->get_ckks_encoder()->slot_count() * 2;
  server_client_relu_packed_test(10, slot_count, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_relu_packed_all_slots) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
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
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  size_t slot_count = he_backend->get_ckks_encoder()->slot_count() * 2;
  server_client_relu_packed_test(10, slot_count, true, true, 7.0);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_bounded_relu_packed_all_slots) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  size_t slot_count = he_backend->get_ckks_encoder()->slot_count();
  server_client_relu_packed_test(10, slot_count, false, true, 8.0);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_relu_packed_845) {
  server_client_relu_packed_test(845, 1, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_relu_packed_1000) {
  server_client_relu_packed_test(1000, 1, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_relu_packed_50000) {
  server_client_relu_packed_test(50000, 1, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_bounded_relu_packed_845) {
  server_client_relu_packed_test(845, 1, true, true, 6.0);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_bounded_relu_packed_1000) {
  server_client_relu_packed_test(1000, 1, true, true, 6.0);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_bounded_relu_packed_50000) {
  server_client_relu_packed_test(50000, 1, true, true, 6.0);
}

auto server_client_maxpool_test = [](const ngraph::Shape& shape,
                                     const ngraph::Shape& window_shape,
                                     const std::vector<float>& input,
                                     const std::vector<float>& output,
                                     const bool arg1_encrypted,
                                     const bool complex_packing,
                                     const bool packed) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        ngraph::he::HESealEncryptionParameters::
            default_complex_packing_parms());
  }

  auto a = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto t = std::make_shared<ngraph::op::MaxPool>(a, window_shape);
  auto f = std::make_shared<ngraph::Function>(t, ngraph::ParameterVector{a});

  std::string tensor_config{"client_input,encrypt"};
  if (packed) {
    tensor_config.append(",packed");
  }
  std::string error_str;
  he_backend->set_config(
      {{"enable_client", "true"}, {a->get_name(), tensor_config}}, error_str);

  // Server inputs which are not used
  auto t_dummy = ngraph::test::he::tensor_from_flags(*he_backend, shape,
                                                     arg1_encrypted, packed);
  auto t_result = ngraph::test::he::tensor_from_flags(
      *he_backend, t->get_shape(), arg1_encrypted, packed);
  size_t batch_size =
      std::static_pointer_cast<ngraph::he::HETensor>(t_dummy)->get_batch_size();

  // Used for dummy server inputs
  float dummy_float = 99;
  copy_data(t_dummy, std::vector<float>(shape_size(shape), dummy_float));

  std::vector<float> results;
  auto client_thread = std::thread([&]() {
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {a->get_name(), make_pair("encrypt", input)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = std::static_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(ngraph::test::he::all_close(results, output));
};

NGRAPH_TEST(${BACKEND_NAME},
            server_client_max_pool_1d_1channel_1image_encrypted_real_unpacked) {
  server_client_maxpool_test(
      ngraph::Shape{1, 1, 14}, ngraph::Shape{3},
      std::vector<float>{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
      std::vector<float>{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}, true, false,
      false);
}

NGRAPH_TEST(${BACKEND_NAME},
            server_client_max_pool_1d_1channel_2image_encrypted_real_packed) {
  server_client_maxpool_test(
      ngraph::Shape{2, 1, 14}, ngraph::Shape{3},
      ngraph::test::NDArray<float, 3>(
          {{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
           {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
          .get_vector(),
      ngraph::test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}},
                                       {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
          .get_vector(),
      true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME},
            server_client_max_pool_1d_2channel_2image_encrypted_real_packed) {
  server_client_maxpool_test(
      ngraph::Shape{2, 2, 14}, ngraph::Shape{3},
      ngraph::test::NDArray<float, 3>(
          {{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
            {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},
           {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
            {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
          .get_vector(),
      ngraph::test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0},
                                        {0, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1}},
                                       {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2},
                                        {2, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2}}})
          .get_vector(),
      true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME},
            server_client_max_pool_2d_2channel_2image_encrypted_real_packed) {
  server_client_maxpool_test(
      ngraph::Shape{2, 2, 5, 5}, ngraph::Shape{2, 3},
      ngraph::test::NDArray<float, 4>({{{{0, 1, 0, 2, 1},  // img 0 chan 0
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
          .get_vector(),
      ngraph::test::NDArray<float, 4>({{{{3, 3, 2},  // img 0 chan 0
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
          .get_vector(),
      true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME},
            server_client_pad_max_pool_1d_1channel_1image_plain) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  size_t batch_size = 1;

  ngraph::Shape shape_a{1, 1, 12};
  ngraph::Shape window_shape{3};
  auto a =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape_a);
  auto b = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{},
                                        std::vector<float>({0}));
  ngraph::CoordinateDiff padding_below{0, 0, 1};
  ngraph::CoordinateDiff padding_above{0, 0, 1};
  auto c =
      std::make_shared<ngraph::op::Pad>(a, b, padding_below, padding_above);
  ngraph::Shape shape_r{1, 1, 12};
  auto f = std::make_shared<ngraph::Function>(
      std::make_shared<ngraph::op::MaxPool>(c, window_shape),
      ngraph::ParameterVector{a});

  std::string error_str;
  he_backend->set_config(
      {{"enable_client", "true"}, {a->get_name(), "client_input,encrypt"}},
      error_str);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(ngraph::element::f32, shape_a);
  auto t_result =
      he_backend->create_cipher_tensor(ngraph::element::f32, shape_r);

  // Used for dummy server inputs
  float dummy_float = 99;
  copy_data(t_dummy, std::vector<float>(shape_size(shape_a), dummy_float));

  std::vector<float> results;
  auto client_thread = std::thread([&]() {
    std::vector<float> inputs{-1, -1, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0};
    auto he_client = ngraph::he::HESealClient(
        "localhost", 34000, batch_size,
        ngraph::he::HETensorConfigMap<float>{
            {a->get_name(), make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = std::static_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(ngraph::test::he::all_close(
      results,
      ngraph::test::NDArray<float, 3>({{{0, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}})
          .get_vector(),
      1e-3f));
}
