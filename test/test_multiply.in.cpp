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

NGRAPH_TEST(${BACKEND_NAME}, multiply_2_3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape{2, 3};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Multiply>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend.get());

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_b = inputs[1];
    auto t_result = results[0];

    copy_data(t_a,
              test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
    copy_data(t_b,
              test::NDArray<float, 2>({{7, 8, 9}, {10, 11, 12}}).get_vector());
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close(
        read_vector<float>(t_result),
        (test::NDArray<float, 2>({{7, 16, 27}, {40, 55, 72}})).get_vector(),
        1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_2_3_halves) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape{2, 3};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Multiply>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend.get());

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_b = inputs[1];
    auto t_result = results[0];

    copy_data(t_a, test::NDArray<float, 2>({{1.5, 2.5, 3.5}, {4.5, 5.5, 6.5}})
                       .get_vector());
    copy_data(t_b,
              test::NDArray<float, 2>({{7.5, 8.5, 9.5}, {10.5, 11.5, 12.5}})
                  .get_vector());
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close(read_vector<float>(t_result),
                          (test::NDArray<float, 2>(
                               {{11.25, 21.25, 33.25}, {47.25, 63.25, 81.25}}))
                              .get_vector(),
                          1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, square_2_3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape{2, 3};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Multiply>(a, a);
  auto f = make_shared<Function>(t, ParameterVector{a});

  auto tensors_list =
      generate_plain_cipher_tensors({t}, {a}, backend.get(), true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_result = results[0];

    copy_data(t_a,
              test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_TRUE(all_close(
        read_vector<float>(t_result),
        (test::NDArray<float, 2>({{1, 4, 9}, {16, 25, 36}})).get_vector(),
        1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_optimized_2_3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape{2, 3};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Multiply>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend.get());

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_b = inputs[1];
    auto t_result = results[0];

    copy_data(t_a,
              test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
    copy_data(t_b,
              test::NDArray<float, 2>({{-1, 0, 1}, {-1, 0, 1}}).get_vector());
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close(
        read_vector<float>(t_result),
        (test::NDArray<float, 2>({{-1, 0, 3}, {-4, 0, 6}})).get_vector(),
        1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_4_3_batch) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape_a{4, 3};
  Shape shape_b{4, 3};
  Shape shape_r{4, 3};
  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto b = make_shared<op::Parameter>(element::f32, shape_b);
  auto t = make_shared<op::Multiply>(a, b);

  auto f = make_shared<Function>(t, ParameterVector{a, b});

  // Create some tensors for input/output
  auto t_a = he_backend->create_batched_plain_tensor(element::f32, shape_a);
  auto t_b = he_backend->create_batched_plain_tensor(element::f32, shape_b);
  auto t_result =
      he_backend->create_batched_plain_tensor(element::f32, shape_r);

  copy_data(t_a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  copy_data(t_b, vector<float>{13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a, t_b});
  EXPECT_TRUE(all_close(
      (vector<float>{13, 28, 45, 64, 85, 108, 133, 160, 189, 220, 253, 288}),
      read_vector<float>(t_result), 1e-3f));
}
