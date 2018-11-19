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

NGRAPH_TEST(${BACKEND_NAME}, reverse_0d) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto r = make_shared<op::Reverse>(A, AxisSet{});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, vector<float>{6});

    backend->call(f, {result}, {a});
    EXPECT_TRUE(all_close((vector<float>{6}), read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_1d_nochange) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{8};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto r = make_shared<op::Reverse>(A, AxisSet{});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7});

    backend->call(f, {result}, {a});
    EXPECT_TRUE(all_close((vector<float>{0, 1, 2, 3, 4, 5, 6, 7}),
                          read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_1d_0) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{8};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto r = make_shared<op::Reverse>(A, AxisSet{0});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7});

    backend->call(f, {result}, {a});
    EXPECT_TRUE(all_close((vector<float>{7, 6, 5, 4, 3, 2, 1, 0}),
                          read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_nochange) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{4, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto r = make_shared<op::Reverse>(A, AxisSet{});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, test::NDArray<float, 2>(
                     {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}})
                     .get_vector());

    backend->call(f, {result}, {a});
    EXPECT_TRUE(all_close(
        (test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}})
             .get_vector()),
        read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_0) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{4, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto r = make_shared<op::Reverse>(A, AxisSet{0});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, test::NDArray<float, 2>(
                     {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}})
                     .get_vector());

    backend->call(f, {result}, {a});
    EXPECT_TRUE(all_close(
        (test::NDArray<float, 2>({{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}})
             .get_vector()),
        read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_1) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{4, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto r = make_shared<op::Reverse>(A, AxisSet{1});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, test::NDArray<float, 2>(
                     {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}})
                     .get_vector());

    backend->call(f, {result}, {a});
    EXPECT_TRUE(all_close(
        (test::NDArray<float, 2>({{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}})
             .get_vector()),
        read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_01) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{4, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto r = make_shared<op::Reverse>(A, AxisSet{0, 1});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, test::NDArray<float, 2>(
                     {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}})
                     .get_vector());

    backend->call(f, {result}, {a});
    EXPECT_TRUE(all_close(
        (test::NDArray<float, 2>({{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}})
             .get_vector()),
        read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_nochange) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{2, 4, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto r = make_shared<op::Reverse>(A, AxisSet{});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, test::NDArray<float, 3>(
                     {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                      {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                     .get_vector());

    backend->call(f, {result}, {a});
    EXPECT_TRUE(all_close(
        (test::NDArray<float, 3>(
             {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
              {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
             .get_vector()),
        read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_0) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{2, 4, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto r = make_shared<op::Reverse>(A, AxisSet{0});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, test::NDArray<float, 3>(
                     {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                      {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                     .get_vector());

    backend->call(f, {result}, {a});
    EXPECT_TRUE(all_close(
        (test::NDArray<float, 3>(
             {{{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}},
              {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}})
             .get_vector()),
        read_vector<float>(result), 1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_1) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{2, 4, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto r = make_shared<op::Reverse>(A, AxisSet{1});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, test::NDArray<float, 3>(
                     {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                      {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                     .get_vector());

    backend->call(f, {result}, {a});
    EXPECT_TRUE(all_close(
        (test::NDArray<float, 3>(
             {{{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}},
              {{21, 22, 23}, {18, 19, 20}, {15, 16, 17}, {12, 13, 14}}})
             .get_vector()),
        read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_2) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{2, 4, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto r = make_shared<op::Reverse>(A, AxisSet{2});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, test::NDArray<float, 3>(
                     {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                      {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                     .get_vector());

    backend->call(f, {result}, {a});
    EXPECT_TRUE(all_close(
        (test::NDArray<float, 3>(
             {{{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}},
              {{14, 13, 12}, {17, 16, 15}, {20, 19, 18}, {23, 22, 21}}})
             .get_vector()),
        read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_01) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{2, 4, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto r = make_shared<op::Reverse>(A, AxisSet{0, 1});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, test::NDArray<float, 3>(
                     {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                      {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                     .get_vector());

    backend->call(f, {result}, {a});
    EXPECT_TRUE(all_close(
        (test::NDArray<float, 3>(
             {{{21, 22, 23}, {18, 19, 20}, {15, 16, 17}, {12, 13, 14}},
              {{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}}})
             .get_vector()),
        read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_02) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{2, 4, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto r = make_shared<op::Reverse>(A, AxisSet{0, 2});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, test::NDArray<float, 3>(
                     {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                      {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                     .get_vector());

    backend->call(f, {result}, {a});
    EXPECT_TRUE(all_close(
        (test::NDArray<float, 3>(
             {{{14, 13, 12}, {17, 16, 15}, {20, 19, 18}, {23, 22, 21}},
              {{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}}})
             .get_vector()),
        read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_12) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{2, 4, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto r = make_shared<op::Reverse>(A, AxisSet{1, 2});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, test::NDArray<float, 3>(
                     {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                      {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                     .get_vector());

    backend->call(f, {result}, {a});
    EXPECT_TRUE(all_close(
        (test::NDArray<float, 3>(
             {{{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}},
              {{23, 22, 21}, {20, 19, 18}, {17, 16, 15}, {14, 13, 12}}})
             .get_vector()),
        read_vector<float>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_012) {
  auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

  Shape shape{2, 4, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto r = make_shared<op::Reverse>(A, AxisSet{0, 1, 2});
  auto f = make_shared<Function>(r, op::ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list = generate_plain_cipher_tensors({r}, {A}, backend, true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, test::NDArray<float, 3>(
                     {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                      {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                     .get_vector());

    backend->call(f, {result}, {a});
    EXPECT_TRUE(all_close(
        (test::NDArray<float, 3>(
             {{{23, 22, 21}, {20, 19, 18}, {17, 16, 15}, {14, 13, 12}},
              {{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}}})
             .get_vector()),
        read_vector<float>(result)));
  }
}
