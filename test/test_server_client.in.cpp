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

NGRAPH_TEST(${BACKEND_NAME}, server_client_create) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  size_t batch_size = 1;

  Shape shape{batch_size, 3};
  auto a = op::Constant::create(element::f32, shape, {0.1, 0.2, 0.3});
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Add>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{b});

  auto t_a = he_backend->create_cipher_tensor(element::f32, shape);
  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, vector<float>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});

  vector<float> inputs{1, 2, 3};
  vector<float> results;
  auto client_thread = std::thread([this, &inputs, &results, &batch_size]() {
    auto he_client =
        ngraph::he::HESealClient("localhost", 34000, batch_size, inputs);

    while (!he_client.is_done()) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    results = he_client.get_results();
    NGRAPH_INFO << "got results size " << results.size();
    for (const auto& elem : results) {
      NGRAPH_INFO << elem;
    }
  });

  auto handle = dynamic_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  NGRAPH_INFO << "enabling client";
  handle->enable_client();

  NGRAPH_INFO << "calling with validate";
  handle->call_with_validate({t_result}, {t_dummy});
  NGRAPH_INFO << "Done with call with validate";

  NGRAPH_INFO << "Cleint results";
  client_thread.join();
  NGRAPH_INFO << "results.size " << results.size();
  for (const auto& elem : results) {
    NGRAPH_INFO << elem;
  }

  EXPECT_TRUE(all_close(results, vector<float>{1.1, 2.2, 3.3}, 1e-3f));
}
