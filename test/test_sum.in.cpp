/*******************************************************************************
 * Copyright 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

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
