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

NGRAPH_TEST(${BACKEND_NAME}, negate_2_3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape{2, 3};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Negative>(a);
  auto f = make_shared<Function>(t, ParameterVector{a});

  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({t}, {a}, backend.get(), true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_result = results[0];

    copy_data(t_a,
              test::NDArray<float, 2>({{-3, -2, -1}, {0, 1, 2}}).get_vector());
    auto handle = backend->compile(f);
    handle->call({t_result}, {t_a});
    EXPECT_TRUE(all_close(
        read_vector<float>(t_result),
        (test::NDArray<float, 2>({{3, 2, 1}, {0, -1, -2}})).get_vector()));
  }
}
