/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*\
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <assert.h>

#include "he_backend.hpp"
#include "seal/he_seal_backend.hpp"

#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

#include "test_util.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

TEST(${BACKEND_NAME}, convolution_2d_1image)
{
    auto shape_a = Shape{1, 1, 5, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto shape_b = Shape{1, 1, 3, 3};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto shape_r = Shape{1, 1, 3, 3};
    auto f = make_shared<Function>(make_shared<op::Convolution>(A, B, Strides{1, 1}, Strides{1, 1}),
                                   op::ParameterVector{A, B});

    // Create some tensors for input/output
    auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_he_data(a,
                 vector<float>{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                               2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
                 backend);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_he_data(b, vector<float>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, backend);
    auto result = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{9, 9, 9, 9, 9, 9, 9, 9, 9};

    backend->call(f, {result}, {a, b});
    EXPECT_TRUE(all_close(vector<float>{expected_result}, read_he_vector<float>(result, backend)));
}

TEST(${BACKEND_NAME}, convolution_2d_1image_2outputs)
{
    auto shape_a = Shape{1, 1, 3, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto shape_b = Shape{2, 1, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto shape_r = Shape{1, 2, 2, 4};
    auto f = make_shared<Function>(make_shared<op::Convolution>(A, B, Strides{1, 1}, Strides{1, 1}),
                                   op::ParameterVector{A, B});

    auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_he_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, backend);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_he_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, backend);
    auto result = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{
        51, 61, 71, 81, 101, 111, 121, 131, 115, 141, 167, 193, 245, 271, 297, 323};

    backend->call(f, {result}, {a, b});
    EXPECT_TRUE(all_close(vector<float>{expected_result}, read_he_vector<float>(result, backend)));
}

TEST(${BACKEND_NAME}, convolution_2d_1item)
{
    Shape shape_a{1, 1, 3, 5};
    Shape shape_b{2, 1, 2, 2};
    Shape shape_r{1, 2, 2, 4};
    auto make_graph = [shape_a, shape_b] {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        auto B = make_shared<op::Parameter>(element::f32, shape_b);
        return make_shared<Function>(
            make_shared<op::Convolution>(A,
                                         B,
                                         Strides{1, 1},        // move_strides
                                         Strides{1, 1},        // filter_dilation
                                         CoordinateDiff{0, 0}, // below_pads
                                         CoordinateDiff{0, 0}, // above_pads
                                         Strides{1, 1}),       // data_dilation
            op::ParameterVector{A, B});
    };

    auto function = make_graph();

    auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_he_data(
        a,
        vector<float>{
            -8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f, -8.f, 5.f, -8.f, 1.f, 2.f, 8.f, -2.f},
        backend);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_he_data(b, vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f}, backend);
    auto result = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{32.0f,
                                  -18.0f,
                                  56.0f,
                                  56.0f,
                                  -42.0f,
                                  -14.0f,
                                  -16.0f,
                                  46.0f,
                                  -54.0f,
                                  -9.0f,
                                  -30.0f,
                                  48.0f,
                                  78.0f,
                                  -33.0f,
                                  -123.0f,
                                  -21.0f};

    backend->call(function, {result}, {a, b});
    EXPECT_TRUE(all_close(vector<float>{expected_result}, read_he_vector<float>(result, backend)));
}

TEST(${BACKEND_NAME}, convolution_2d_1item_padded_1_1x1_1)
{
    Shape shape_a{1, 1, 3, 5};
    Shape shape_b{2, 1, 2, 2};
    Shape shape_r{1, 2, 4, 6};
    auto make_graph = [shape_a, shape_b] {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        auto B = make_shared<op::Parameter>(element::f32, shape_b);
        return make_shared<Function>(
            make_shared<op::Convolution>(A,
                                         B,
                                         Strides{1, 1},        // move_strides
                                         Strides{1, 1},        // filter_dilation
                                         CoordinateDiff{1, 1}, // below_pads
                                         CoordinateDiff{1, 1}, // above_pads
                                         Strides{1, 1}),       // data_dilation
            op::ParameterVector{A, B});
    };

    auto function = make_graph();

    auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_he_data(
        a,
        vector<float>{
            -8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f, -8.f, 5.f, -8.f, 1.f, 2.f, 8.f, -2.f},
        backend);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_he_data(b, vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f}, backend);
    auto result = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{16.0f,  28.0f,  0.0f,   20.0f,  -10.0f, -36.0f, -34.0f, 32.0f,
                                  -18.0f, 56.0f,  56.0f,  -92.0f, 34.0f,  -42.0f, -14.0f, -16.0f,
                                  46.0f,  -32.0f, -16.0f, 66.0f,  -4.0f,  0.0f,   -68.0f, 16.0f,
                                  24.0f,  -6.0f,  12.0f,  6.0f,   -27.0f, 0.0f,   -99.0f, -54.0f,
                                  -9.0f,  -30.0f, 48.0f,  81.0f,  105.0f, 78.0f,  -33.0f, -123.0f,
                                  -21.0f, 45.0f,  -72.0f, -63.0f, 27.0f,  90.0f,  54.0f,  -18.0f};

    backend->call(function, {result}, {a, b});
    EXPECT_TRUE(
        all_close(vector<float>{expected_result}, read_he_vector<float>(result, backend), 1e-5f));
}

TEST(${BACKEND_NAME}, convolution_2d_1item_padded_2_3x4_5)
{
    Shape shape_a{1, 1, 3, 5};
    Shape shape_b{2, 1, 2, 2};
    Shape shape_r{1, 2, 8, 12};
    auto make_graph = [shape_a, shape_b] {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        auto B = make_shared<op::Parameter>(element::f32, shape_b);
        return make_shared<Function>(
            make_shared<op::Convolution>(A,
                                         B,
                                         Strides{1, 1},        // move_strides
                                         Strides{1, 1},        // filter_dilation
                                         CoordinateDiff{2, 3}, // below_pads
                                         CoordinateDiff{4, 5}, // above_pads
                                         Strides{1, 1}),       // data_dilation
            op::ParameterVector{A, B});
    };

    auto function = make_graph();

    // Create some tensors for input/output
    auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_he_data(
        a,
        vector<float>{
            -8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f, -8.f, 5.f, -8.f, 1.f, 2.f, 8.f, -2.f},
        backend);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_he_data(b, vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f}, backend);
    auto result = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{
        0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,    0.0f,   0.0f,   0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 16.0f,  28.0f,  0.0f,   20.0f,   -10.0f, -36.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -34.0f, 32.0f,  -18.0f, 56.0f,   56.0f,  -92.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 34.0f,  -42.0f, -14.0f, -16.0f,  46.0f,  -32.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -16.0f, 66.0f,  -4.0f,  0.0f,    -68.0f, 16.0f,  0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,    0.0f,   0.0f,   0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,    0.0f,   0.0f,   0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,    0.0f,   0.0f,   0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,    0.0f,   0.0f,   0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 24.0f,  -6.0f,  12.0f,  6.0f,    -27.0f, 0.0f,   0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -99.0f, -54.0f, -9.0f,  -30.0f,  48.0f,  81.0f,  0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 105.0f, 78.0f,  -33.0f, -123.0f, -21.0f, 45.0f,  0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -72.0f, -63.0f, 27.0f,  90.0f,   54.0f,  -18.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,    0.0f,   0.0f,   0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,    0.0f,   0.0f,   0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,    0.0f,   0.0f,   0.0f, 0.0f, 0.0f, 0.0f};

    backend->call(function, {result}, {a, b});
    EXPECT_TRUE(
        all_close(vector<float>{expected_result}, read_he_vector<float>(result, backend), 1e-5f));
}

TEST(${BACKEND_NAME}, convolution_2d_2items)
{
    Shape shape_a{2, 1, 3, 5};
    Shape shape_b{2, 1, 2, 2};
    Shape shape_r{2, 2, 2, 4};
    auto make_graph = [shape_a, shape_b] {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        auto B = make_shared<op::Parameter>(element::f32, shape_b);
        return make_shared<Function>(
            make_shared<op::Convolution>(A,
                                         B,
                                         Strides{1, 1},        // move_strides
                                         Strides{1, 1},        // filter_dilation
                                         CoordinateDiff{0, 0}, // below_pads
                                         CoordinateDiff{0, 0}, // above_pads
                                         Strides{1, 1}),       // data_dilation
            op::ParameterVector{A, B});
    };

    auto function = make_graph();

    // Create some tensors for input/output
    auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_he_data(a,
                 vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                               -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                               6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
                 backend);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_he_data(b, vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f}, backend);
    auto result = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{32.0f,  -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,  46.0f,
                                  -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f, -123.0f, -21.0f,
                                  -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f, -10.0f, 8.0f,    64.0f,
                                  138.0f, 30.0f,  -30.0f, 6.0f,   48.0f,  -66.0f, -42.0f,  72.0f};

    backend->call(function, {result}, {a, b});
    EXPECT_TRUE(all_close(vector<float>{expected_result}, read_he_vector<float>(result, backend)));
}

TEST(${BACKEND_NAME}, convolution_2d_2items_strided_padded)
{
    Shape shape_a{2, 1, 3, 5};
    Shape shape_b{2, 1, 2, 2};
    Shape shape_r{2, 2, 6, 7};
    auto make_graph = [shape_a, shape_b] {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        auto B = make_shared<op::Parameter>(element::f32, shape_b);
        return make_shared<Function>(
            make_shared<op::Convolution>(A,
                                         B,
                                         Strides{2, 2},        // move_strides
                                         Strides{1, 1},        // filter_dilation
                                         CoordinateDiff{4, 2}, // below_pads
                                         CoordinateDiff{5, 7}, // above_pads
                                         Strides{1, 1}),       // data_dilation
            op::ParameterVector{A, B});
    };

    auto function = make_graph();

    // Create some tensors for input/output
    auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_he_data(a,
                 vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                               -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                               6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
                 backend);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_he_data(b, vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f}, backend);
    auto result = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{
        0.0f,  0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,   0.0f, 0.0f, 0.0f,   0.0f,   0.0f,
        0.0f,  0.0f, 0.0f, 32.0f,  56.0f,  -92.0f, 0.0f,   0.0f, 0.0f, 0.0f,   66.0f,  0.0f,
        16.0f, 0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,   0.0f, 0.0f, 0.0f,   0.0f,   0.0f,
        0.0f,  0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,   0.0f, 0.0f, 0.0f,   0.0f,   0.0f,
        0.0f,  0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,   0.0f, 0.0f, -54.0f, -30.0f, 81.0f,
        0.0f,  0.0f, 0.0f, 0.0f,   -63.0f, 90.0f,  -18.0f, 0.0f, 0.0f, 0.0f,   0.0f,   0.0f,
        0.0f,  0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,   0.0f, 0.0f, 0.0f,   0.0f,   0.0f,
        0.0f,  0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,   0.0f, 0.0f, 0.0f,   0.0f,   0.0f,
        0.0f,  0.0f, 0.0f, -52.0f, 82.0f,  -28.0f, 0.0f,   0.0f, 0.0f, 0.0f,   -2.0f,  -64.0f,
        72.0f, 0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,   0.0f, 0.0f, 0.0f,   0.0f,   0.0f,
        0.0f,  0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,   0.0f, 0.0f, 0.0f,   0.0f,   0.0f,
        0.0f,  0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,   0.0f, 0.0f, 138.0f, -30.0f, 0.0f,
        0.0f,  0.0f, 0.0f, 0.0f,   -9.0f,  27.0f,  -81.0f, 0.0f, 0.0f, 0.0f,   0.0f,   0.0f,
        0.0f,  0.0f, 0.0f, 0.0f,   0.0f,   0.0f,   0.0f,   0.0f, 0.0f, 0.0f,   0.0f,   0.0f};

    backend->call(function, {result}, {a, b});
    EXPECT_TRUE(
        all_close(vector<float>{expected_result}, read_he_vector<float>(result, backend), 1e-5f));
}
