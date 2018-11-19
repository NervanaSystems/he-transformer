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

NGRAPH_TEST(${BACKEND_NAME}, constant) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
  Shape shape{2, 2};
  auto A = op::Constant::create(element::f32, shape, {0.1, 0.2, 0.3, 0.4});
  auto f = make_shared<Function>(A, op::ParameterVector{});

  auto result = backend->create_tensor(element::f32, shape);
  backend->call(f, {result}, {});
  EXPECT_TRUE(all_close((vector<float>{0.1, 0.2, 0.3, 0.4}),
                        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, constant_abc) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
  Shape shape{2, 2};
  auto A = op::Constant::create(element::f32, shape, {1, 2, 3, 4});
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto C = make_shared<op::Parameter>(element::f32, shape);
  auto t = (A + B) * C;
  auto f = make_shared<Function>(t, op::ParameterVector{B, C});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {B, C}, backend);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto b = inputs[0];
    auto c = inputs[1];
    auto result = results[0];

    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    backend->call(f, {result}, {b, c});

    EXPECT_TRUE(all_close(
        read_vector<float>(result),
        (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector()));
  }
}