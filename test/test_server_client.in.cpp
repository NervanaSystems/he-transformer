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

#include "ngraph/ngraph.hpp"
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

NGRAPH_TEST(${BACKEND_NAME}, server_client_add_3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  Shape shape{batch_size, 3};
  auto a = op::Constant::create(element::f32, shape, {0.1, 0.2, 0.3});
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Add>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{b});

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});

  vector<float> inputs{1, 2, 3};
  vector<float> results;
  auto client_thread = std::thread([&inputs, &results, &batch_size]() {
    auto he_client =
        ngraph::he::HESealClient("localhost", 34000, batch_size, inputs);

    while (!he_client.is_done()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = dynamic_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));
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

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});

  vector<float> inputs{-1, -0.2, 3};
  vector<float> results;
  auto client_thread = std::thread([&inputs, &results, &batch_size]() {
    auto he_client =
        ngraph::he::HESealClient("localhost", 34000, batch_size, inputs);

    while (!he_client.is_done()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = dynamic_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));
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

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f64, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f64, shape);

  // Used for dummy server inputs
  double DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<double>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});

  vector<double> inputs{-1, -0.2, 3};
  vector<double> results;
  auto client_thread = std::thread([&inputs, &results, &batch_size]() {
    auto he_client =
        ngraph::he::HESealClient("localhost", 34000, batch_size, inputs);

    while (!he_client.is_done()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto double_results = he_client.get_results();
    results = std::vector<double>(double_results.begin(), double_results.end());
  });

  auto handle = dynamic_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));
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

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::i64, shape);
  auto t_result = he_backend->create_cipher_tensor(element::i64, shape);

  // Used for dummy server inputs
  double DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<double>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});

  vector<double> inputs{-1, 0, 3};
  vector<int64_t> results;
  auto client_thread = std::thread([&inputs, &results, &batch_size]() {
    auto he_client =
        ngraph::he::HESealClient("localhost", 34000, batch_size, inputs);

    while (!he_client.is_done()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto double_results = he_client.get_results();
    results.resize(double_results.size());
    for (size_t i = 0; i < results.size(); ++i) {
      results[i] = std::round(double_results[i]);
    }
  });

  auto handle = dynamic_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));
  handle->enable_client();
  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(all_close(results, vector<int64_t>{0, 2, 6}, 0L));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_relu) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  Shape shape{batch_size, 3};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto relu = make_shared<op::Relu>(a);
  auto f = make_shared<Function>(relu, ParameterVector{a});

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});

  vector<float> inputs{-1, -0.2, 3};
  vector<float> results;
  auto client_thread = std::thread([&inputs, &results, &batch_size]() {
    auto he_client =
        ngraph::he::HESealClient("localhost", 34000, batch_size, inputs);

    while (!he_client.is_done()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = dynamic_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));
  handle->enable_client();
  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(all_close(results, vector<float>{0, 0, 3}, 1e-3f));
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

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, result_shape);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});

  vector<float> inputs{-1, -0.2, 3};
  vector<float> results;
  auto client_thread = std::thread([&inputs, &results, &batch_size]() {
    auto he_client =
        ngraph::he::HESealClient("localhost", 34000, batch_size, inputs);

    while (!he_client.is_done()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = dynamic_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));
  handle->enable_client();
  NGRAPH_INFO << "Calling with validate";
  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(all_close(results, vector<float>{0, 0, 3}, 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_relu_packed) {
  auto tmp_backend = runtime::Backend::create("${BACKEND_NAME}");
  auto tmp_he_backend =
      static_cast<ngraph::he::HESealBackend*>(tmp_backend.get());
  vector<size_t> batch_sizes{
      1, 2, 3,
      tmp_he_backend->get_encryption_parameters().poly_modulus_degree()};

  for (auto batch_size : batch_sizes) {
    for (auto complex_packing : vector<bool>{true, false}) {
      auto backend = runtime::Backend::create("${BACKEND_NAME}");
      auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
      he_backend->complex_packing() = complex_packing;

      if (complex_packing &&
          batch_size ==
              he_backend->get_encryption_parameters().poly_modulus_degree()) {
        batch_size /= 2;
      }

      Shape shape{batch_size, 3};
      auto a = make_shared<op::Parameter>(element::f32, shape);
      auto relu = make_shared<op::Relu>(a);
      auto f = make_shared<Function>(relu, ParameterVector{a});

      // Server inputs which are not used
      auto t_dummy =
          he_backend->create_packed_plain_tensor(element::f32, shape);
      auto t_result =
          he_backend->create_packed_cipher_tensor(element::f32, shape);

      // Used for dummy server inputs
      float DUMMY_FLOAT = 99;
      copy_data(t_dummy, vector<float>(shape_size(shape), DUMMY_FLOAT));

      vector<float> inputs(shape_size(shape));
      vector<float> exp_results(shape_size(shape));
      for (size_t i = 0; i < shape_size(shape); ++i) {
        inputs[i] =
            static_cast<int>(i) - static_cast<int>(shape_size(shape)) / 2;
        exp_results[i] = inputs[i] > 0 ? inputs[i] : 0;
      }

      vector<float> results;
      auto client_thread = std::thread([&]() {
        auto he_client = ngraph::he::HESealClient(
            "localhost", 34000, batch_size, inputs, complex_packing);

        while (!he_client.is_done()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        auto double_results = he_client.get_results();
        results =
            std::vector<float>(double_results.begin(), double_results.end());
      });

      auto handle = dynamic_pointer_cast<ngraph::he::HESealExecutable>(
          he_backend->compile(f));
      handle->enable_client();
      handle->call_with_validate({t_result}, {t_dummy});

      client_thread.join();
      EXPECT_TRUE(all_close(results, exp_results));
    }
  }
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

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape_a);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape_r);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>(shape_size(shape_a), DUMMY_FLOAT));

  vector<float> inputs{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0};
  vector<float> results;
  auto client_thread = std::thread([&inputs, &results, &batch_size]() {
    auto he_client =
        ngraph::he::HESealClient("localhost", 34000, batch_size, inputs);

    while (!he_client.is_done()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = dynamic_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));
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

  // Server inputs which are not used
  auto t_dummy = he_backend->create_packed_plain_tensor(element::f32, shape_a);
  auto t_result =
      he_backend->create_packed_cipher_tensor(element::f32, shape_r);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>(shape_size(shape_a), DUMMY_FLOAT));

  vector<float> inputs =
      test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                               {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
          .get_vector();
  vector<float> results;
  auto client_thread = std::thread([&inputs, &results, &batch_size]() {
    auto he_client =
        ngraph::he::HESealClient("localhost", 34000, batch_size, inputs);

    while (!he_client.is_done()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = dynamic_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));
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

  // Server inputs which are not used
  auto t_dummy = he_backend->create_packed_plain_tensor(element::f32, shape_a);
  auto t_result =
      he_backend->create_packed_cipher_tensor(element::f32, shape_r);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>(shape_size(shape_a), DUMMY_FLOAT));

  vector<float> inputs =
      test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},
                               {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
          .get_vector();
  vector<float> results;
  auto client_thread = std::thread([&inputs, &results, &batch_size]() {
    auto he_client =
        ngraph::he::HESealClient("localhost", 34000, batch_size, inputs);

    while (!he_client.is_done()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = dynamic_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));
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

  // Server inputs which are not used
  auto t_dummy = he_backend->create_packed_plain_tensor(element::f32, shape_a);
  auto t_result =
      he_backend->create_packed_cipher_tensor(element::f32, shape_r);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>(shape_size(shape_a), DUMMY_FLOAT));

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
  vector<float> results;
  auto client_thread = std::thread([&inputs, &results, &batch_size]() {
    auto he_client =
        ngraph::he::HESealClient("localhost", 34000, batch_size, inputs);

    while (!he_client.is_done()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = dynamic_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));
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