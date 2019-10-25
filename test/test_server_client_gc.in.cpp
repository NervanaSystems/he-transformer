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
using namespace ngraph::he;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_add_3_multiple_parameters_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  std::string error_str;
  he_backend->set_config(
      std::map<std::string, std::string>{{"enable_client", "true"},
                                         {"enable_gc", "true"}},
      error_str);

  size_t batch_size = 1;

  Shape shape{batch_size, 3};
  Shape shape_c{3, 1};
  auto a = op::Constant::create(element::f32, shape, {0.1, 0.2, 0.3});
  auto b = make_shared<op::Parameter>(element::f32, shape);
  b->set_op_annotations(
      HEOpAnnotations::client_plaintext_unpacked_annotation());

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
  auto client_thread = thread([&]() {
    vector<float> inputs{1, 2, 3};
    auto he_client = HESealClient(
        "localhost", 34000, batch_size,
        HETensorConfigMap<float>{{b->get_name(), make_pair("plain", inputs)}});

    auto double_results = he_client.get_results();
    results = vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = static_pointer_cast<HESealExecutable>(he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy, t_c});
  client_thread.join();
  EXPECT_TRUE(
      test::he::all_close(results, vector<float>{5.1, 7.2, 9.3}, 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_add_3_relu) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  std::string error_str;
  he_backend->set_config(
      std::map<std::string, std::string>{{"enable_client", "true"},
                                         {"enable_gc", "true"}},
      error_str);

  size_t batch_size = 1;

  Shape shape{batch_size, 3};
  auto a = op::Constant::create(element::f32, shape, {0.1, 0.2, 0.3});
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Add>(a, b);
  auto relu = make_shared<op::Relu>(t);
  auto f = make_shared<Function>(relu, ParameterVector{b});

  b->set_op_annotations(
      HEOpAnnotations::client_ciphertext_unpacked_annotation());

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});

  vector<float> results;
  auto client_thread = thread([&]() {
    vector<float> inputs{-1, -0.2, 3};
    auto he_client =
        HESealClient("localhost", 34000, batch_size,
                     HETensorConfigMap<float>{
                         {b->get_name(), make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = vector<float>(double_results.begin(), double_results.end());
  });

  auto handle = static_pointer_cast<HESealExecutable>(he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(test::he::all_close(results, vector<float>{0, 0, 3.3}, 1e-3f));
}
