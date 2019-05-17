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

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_1channel_1image) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  Shape shape_a{1, 1, 14};
  Shape window_shape{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{1, 1, 12};

  float denom = 3.0;

  auto t = make_shared<op::AvgPool>(A, window_shape);
  auto f = make_shared<Function>(t, ParameterVector{A});

  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({t}, {A}, backend.get(), true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(
        a, test::NDArray<float, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}
               .get_vector());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(all_close(test::NDArray<float, 3>(
                              {{{1 / denom, 3 / denom, 3 / denom, 3 / denom,
                                 4 / denom, 5 / denom, 5 / denom, 2 / denom,
                                 2 / denom, 2 / denom, 2 / denom, 0 / denom}}})
                              .get_vector(),
                          read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_1channel_2image) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  Shape shape_a{2, 1, 14};
  Shape window_shape{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 1, 12};

  float denom = 3.0;

  auto t = make_shared<op::AvgPool>(A, window_shape);
  auto f = make_shared<Function>(t, ParameterVector{A});

  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({t}, {A}, backend.get(), true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, test::NDArray<float, 3>(
                     {{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                      {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
                     .get_vector());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(all_close(test::NDArray<float, 3>(
                              {{{1 / denom, 3 / denom, 3 / denom, 3 / denom,
                                 4 / denom, 5 / denom, 5 / denom, 2 / denom,
                                 2 / denom, 2 / denom, 2 / denom, 0 / denom}},
                               {{3 / denom, 4 / denom, 2 / denom, 1 / denom,
                                 0 / denom, 2 / denom, 2 / denom, 3 / denom,
                                 1 / denom, 1 / denom, 1 / denom, 3 / denom}}})
                              .get_vector(),
                          read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_1channel_2image_batched) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());
  Shape shape_a{2, 1, 14};
  Shape window_shape{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 1, 12};

  float denom = 3.0;

  auto t = make_shared<op::AvgPool>(A, window_shape);
  auto f = make_shared<Function>(t, ParameterVector{A});

  // Create some tensors for input/output
  auto t_a = he_backend->create_batched_cipher_tensor(element::f32, shape_a);
  auto t_result =
      he_backend->create_batched_cipher_tensor(element::f32, shape_r);

  copy_data(t_a, test::NDArray<float, 3>(
                     {{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                      {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
                     .get_vector());

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a});
  EXPECT_TRUE(all_close(
      test::NDArray<float, 3>(
          {{{1 / denom, 3 / denom, 3 / denom, 3 / denom, 4 / denom, 5 / denom,
             5 / denom, 2 / denom, 2 / denom, 2 / denom, 2 / denom, 0 / denom}},
           {{3 / denom, 4 / denom, 2 / denom, 1 / denom, 0 / denom, 2 / denom,
             2 / denom, 3 / denom, 1 / denom, 1 / denom, 1 / denom,
             3 / denom}}})
          .get_vector(),
      read_vector<float>(t_result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_2channel_2image) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  Shape shape_a{2, 2, 14};
  Shape window_shape{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 2, 12};

  float denom = 3.0;

  auto t = make_shared<op::AvgPool>(A, window_shape);
  auto f = make_shared<Function>(t, ParameterVector{A});

  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({t}, {A}, backend.get(), true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, test::NDArray<float, 3>(
                     {{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                       {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                      {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                       {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
                     .get_vector());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(all_close(test::NDArray<float, 3>(
                              {{{1 / denom, 3 / denom, 3 / denom, 3 / denom,
                                 4 / denom, 5 / denom, 5 / denom, 2 / denom,
                                 2 / denom, 2 / denom, 2 / denom, 0 / denom},
                                {0 / denom, 2 / denom, 2 / denom, 2 / denom,
                                 2 / denom, 5 / denom, 5 / denom, 4 / denom,
                                 3 / denom, 3 / denom, 3 / denom, 1 / denom}},

                               {{3 / denom, 4 / denom, 2 / denom, 1 / denom,
                                 0 / denom, 2 / denom, 2 / denom, 3 / denom,
                                 1 / denom, 1 / denom, 1 / denom, 3 / denom},
                                {3 / denom, 1 / denom, 1 / denom, 1 / denom,
                                 3 / denom, 2 / denom, 2 / denom, 0 / denom,
                                 1 / denom, 2 / denom, 4 / denom, 3 / denom}}})
                              .get_vector(),
                          read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  Shape shape_a{2, 2, 5, 5};
  Shape window_shape{2, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 2, 4, 3};

  float denom = 2 * 3;

  auto t = make_shared<op::AvgPool>(A, window_shape);
  auto f = make_shared<Function>(t, ParameterVector{A});

  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({t}, {A}, backend.get(), true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, test::NDArray<float, 4>({{{{0, 1, 0, 2, 1},  // img 0 chan 0
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
                     .get_vector());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(
        all_close(test::NDArray<float, 4>(
                      {{{{6 / denom, 8 / denom, 5 / denom},  // img 0 chan 0
                         {7 / denom, 5 / denom, 3 / denom},
                         {5 / denom, 2 / denom, 5 / denom},
                         {6 / denom, 5 / denom, 5 / denom}},

                        {{5 / denom, 7 / denom, 6 / denom},  // img 0 chan 1
                         {8 / denom, 6 / denom, 7 / denom},
                         {7 / denom, 2 / denom, 3 / denom},
                         {6 / denom, 1 / denom, 0 / denom}}},

                       {{{5 / denom, 6 / denom, 5 / denom},  // img 1 chan 0
                         {3 / denom, 5 / denom, 9 / denom},
                         {3 / denom, 6 / denom, 9 / denom},
                         {2 / denom, 3 / denom, 3 / denom}},

                        {{5 / denom, 3 / denom, 1 / denom},  // img 1 chan 1
                         {6 / denom, 5 / denom, 4 / denom},
                         {7 / denom, 5 / denom, 6 / denom},
                         {4 / denom, 2 / denom, 4 / denom}}}})
                      .get_vector(),
                  read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_1channel_1image_strided) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  Shape shape_a{1, 1, 8, 8};
  Shape window_shape{2, 3};
  auto window_movement_strides = Strides{3, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{1, 1, 3, 3};

  float denom = 2 * 3;

  auto t = make_shared<op::AvgPool>(A, window_shape, window_movement_strides);
  auto f = make_shared<Function>(t, ParameterVector{A});

  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({t}, {A}, backend.get(), true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, test::NDArray<float, 4>({{{{0, 1, 0, 2, 1, 2, 0, 0},
                                            {0, 3, 2, 0, 0, 0, 1, 0},
                                            {2, 0, 0, 0, 1, 0, 0, 0},
                                            {2, 0, 1, 1, 2, 2, 3, 0},
                                            {0, 2, 1, 0, 0, 0, 1, 0},
                                            {2, 0, 3, 1, 0, 0, 0, 0},
                                            {1, 2, 0, 0, 0, 1, 2, 0},
                                            {1, 0, 2, 0, 0, 0, 1, 0}}}})
                     .get_vector());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(all_close(
        test::NDArray<float, 4>({{{{6 / denom, 5 / denom, 4 / denom},
                                   {6 / denom, 5 / denom, 8 / denom},
                                   {6 / denom, 2 / denom, 4 / denom}}}})
            .get_vector(),
        read_vector<float>(result)));
  }
}
