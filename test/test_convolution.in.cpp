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
#include "seal/he_seal_backend.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_1image) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  auto shape_a = Shape{1, 1, 5, 5};
  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto shape_b = Shape{1, 1, 3, 3};
  auto b = make_shared<op::Parameter>(element::f32, shape_b);
  auto t = make_shared<op::Convolution>(a, b, Strides{1, 1}, Strides{1, 1});
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend.get());

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_b = inputs[1];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0});
    copy_data(t_b, vector<float>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5});

    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close(read_vector<float>(t_result),
                          vector<float>{9, 9, 9, 9, 9, 9, 9, 9, 9}, 1e-1f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_1image_2outputs) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  auto shape_a = Shape{1, 1, 3, 5};
  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto shape_b = Shape{2, 1, 2, 2};
  auto b = make_shared<op::Parameter>(element::f32, shape_b);
  auto t = make_shared<op::Convolution>(a, b, Strides{1, 1}, Strides{1, 1});
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend.get());
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_b = inputs[1];
    auto t_result = results[0];

    copy_data(t_a,
              vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    copy_data(t_b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});

    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close(read_vector<float>(t_result),
                          vector<float>{51, 61, 71, 81, 101, 111, 121, 131, 115,
                                        141, 167, 193, 245, 271, 297, 323},
                          1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_1item) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape_a{1, 1, 3, 5};
  Shape shape_b{2, 1, 2, 2};
  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto b = make_shared<op::Parameter>(element::f32, shape_b);
  auto t = make_shared<op::Convolution>(a, b, Strides{1, 1},  // move_strides
                                        Strides{1, 1},        // filter_dilation
                                        CoordinateDiff{0, 0},  // below_pads
                                        CoordinateDiff{0, 0},  // above_pads
                                        Strides{1, 1});        // data_dilation
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend.get());

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_b = inputs[1];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f,
                                 -8.f, 5.f, -8.f, 1.f, 2.f, 8.f, -2.f});
    copy_data(t_b, vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(
        all_close(read_vector<float>(t_result),
                  vector<float>{32.0f, -18.0f, 56.0f, 56.0f, -42.0f, -14.0f,
                                -16.0f, 46.0f, -54.0f, -9.0f, -30.0f, 48.0f,
                                78.0f, -33.0f, -123.0f, -21.0f},
                  1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_1item_padded_1_1x1_1) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape_a{1, 1, 3, 5};
  Shape shape_b{2, 1, 2, 2};
  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto b = make_shared<op::Parameter>(element::f32, shape_b);
  auto t = make_shared<op::Convolution>(a, b, Strides{1, 1},  // move_strides
                                        Strides{1, 1},        // filter_dilation
                                        CoordinateDiff{1, 1},  // below_pads
                                        CoordinateDiff{1, 1},  // above_pads
                                        Strides{1, 1});        // data_dilation
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend.get());

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_b = inputs[1];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f,
                                 -8.f, 5.f, -8.f, 1.f, 2.f, 8.f, -2.f});
    copy_data(t_b, vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close(
        read_vector<float>(t_result),
        vector<float>{16.0f,  28.0f,  0.0f,   20.0f,  -10.0f,  -36.0f, -34.0f,
                      32.0f,  -18.0f, 56.0f,  56.0f,  -92.0f,  34.0f,  -42.0f,
                      -14.0f, -16.0f, 46.0f,  -32.0f, -16.0f,  66.0f,  -4.0f,
                      0.0f,   -68.0f, 16.0f,  24.0f,  -6.0f,   12.0f,  6.0f,
                      -27.0f, 0.0f,   -99.0f, -54.0f, -9.0f,   -30.0f, 48.0f,
                      81.0f,  105.0f, 78.0f,  -33.0f, -123.0f, -21.0f, 45.0f,
                      -72.0f, -63.0f, 27.0f,  90.0f,  54.0f,   -18.0f},
        1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_1item_padded_2_3x4_5) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape_a{1, 1, 3, 5};
  Shape shape_b{2, 1, 2, 2};
  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto b = make_shared<op::Parameter>(element::f32, shape_b);
  auto t = make_shared<op::Convolution>(a, b, Strides{1, 1},  // move_strides
                                        Strides{1, 1},        // filter_dilation
                                        CoordinateDiff{2, 3},  // below_pads
                                        CoordinateDiff{4, 5},  // above_pads
                                        Strides{1, 1});        // data_dilation
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend.get());

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_b = inputs[1];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f,
                                 -8.f, 5.f, -8.f, 1.f, 2.f, 8.f, -2.f});
    copy_data(t_b, vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close(
        read_vector<float>(t_result),
        vector<float>{
            0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   16.0f,  28.0f,
            0.0f,   20.0f,   -10.0f, -36.0f, 0.0f,   0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,    -34.0f, 32.0f,  -18.0f, 56.0f,  56.0f,  -92.0f,
            0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   34.0f,  -42.0f,
            -14.0f, -16.0f,  46.0f,  -32.0f, 0.0f,   0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,    -16.0f, 66.0f,  -4.0f,  0.0f,   -68.0f, 16.0f,
            0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   24.0f,  -6.0f,
            12.0f,  6.0f,    -27.0f, 0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,    -99.0f, -54.0f, -9.0f,  -30.0f, 48.0f,  81.0f,
            0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   105.0f, 78.0f,
            -33.0f, -123.0f, -21.0f, 45.0f,  0.0f,   0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,    -72.0f, -63.0f, 27.0f,  90.0f,  54.0f,  -18.0f,
            0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f},
        1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_2items) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape_a{2, 1, 3, 5};
  Shape shape_b{2, 1, 2, 2};
  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto b = make_shared<op::Parameter>(element::f32, shape_b);
  auto t = make_shared<op::Convolution>(a, b, Strides{1, 1},  // move_strides
                                        Strides{1, 1},        // filter_dilation
                                        CoordinateDiff{0, 0},  // below_pads
                                        CoordinateDiff{0, 0},  // above_pads
                                        Strides{1, 1});        // data_dilation
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend.get());

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_b = inputs[1];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f,
                                 -8.f, 5.f,  -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,
                                 9.f,  -7.f, 3.f,  0.f,  6.f,  -1.f, -4.f, -2.f,
                                 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f});
    copy_data(t_b, vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close(
        read_vector<float>(t_result),
        vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                      46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                      -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                      -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                      48.0f,   -66.0f, -42.0f, 72.0f},
        1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_2items_strided_padded) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape_a{2, 1, 3, 5};
  Shape shape_b{2, 1, 2, 2};
  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto b = make_shared<op::Parameter>(element::f32, shape_b);
  auto t = make_shared<op::Convolution>(a, b, Strides{2, 2},  // move_strides
                                        Strides{1, 1},        // filter_dilation
                                        CoordinateDiff{4, 2},  // below_pads
                                        CoordinateDiff{5, 7},  // above_pads
                                        Strides{1, 1});        // data_dilation
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend.get());

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_b = inputs[1];
    auto t_result = results[0];

    copy_data(t_a, vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f,
                                 -8.f, 5.f,  -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,
                                 9.f,  -7.f, 3.f,  0.f,  6.f,  -1.f, -4.f, -2.f,
                                 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f});
    copy_data(t_b, vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close(
        read_vector<float>(t_result),
        vector<float>{
            0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   32.0f,
            56.0f,  -92.0f, 0.0f,   0.0f,   0.0f,  0.0f,   66.0f,  0.0f,
            16.0f,  0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
            0.0f,   -54.0f, -30.0f, 81.0f,  0.0f,  0.0f,   0.0f,   0.0f,
            -63.0f, 90.0f,  -18.0f, 0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,   0.0f,   -52.0f, 82.0f, -28.0f, 0.0f,   0.0f,
            0.0f,   0.0f,   -2.0f,  -64.0f, 72.0f, 0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  138.0f, -30.0f, 0.0f,
            0.0f,   0.0f,   0.0f,   0.0f,   -9.0f, 27.0f,  -81.0f, 0.0f,
            0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
            0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f},
        1e-3f));
  }
}
