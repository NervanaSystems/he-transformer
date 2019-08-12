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

#include "ngraph/ngraph.hpp"
#include "seal/he_seal_backend.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_1channel_1image_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  Shape shape_a{1, 1, 14};
  Shape window_shape{3};
  Shape shape_r{1, 1, 12};
  {
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape),
                                   ParameterVector{A});
    auto a = he_backend->create_plain_tensor(element::f32, shape_a);
    copy_data(
        a, test::NDArray<float, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}
               .get_vector());
    auto result = he_backend->create_plain_tensor(element::f32, shape_r);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(all_close(
        (test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}})
             .get_vector()),
        read_vector<float>(result), 1e-3f));
  }
  {
    auto A = make_shared<op::Parameter>(element::f64, shape_a);
    auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape),
                                   ParameterVector{A});
    auto a = he_backend->create_plain_tensor(element::f64, shape_a);
    copy_data(
        a,
        test::NDArray<double, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}
            .get_vector());
    auto result = he_backend->create_plain_tensor(element::f64, shape_r);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(all_close(
        (test::NDArray<double, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}})
             .get_vector()),
        read_vector<double>(result), 1e-3));
  }
  {
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape),
                                   ParameterVector{A});
    auto a = he_backend->create_plain_tensor(element::i64, shape_a);
    copy_data(
        a,
        test::NDArray<int64_t, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}
            .get_vector());
    auto result = he_backend->create_plain_tensor(element::i64, shape_r);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(all_close(
        (test::NDArray<int64_t, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}})
             .get_vector()),
        read_vector<int64_t>(result), 0L));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_1channel_1image_cipher) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape_a{1, 1, 14};
  Shape window_shape{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{1, 1, 12};
  auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape),
                                 ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  copy_data(
      a, test::NDArray<float, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}
             .get_vector());
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  auto handle = backend->compile(f);
  handle->call_with_validate({result}, {a});
  EXPECT_TRUE(all_close(
      (test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}})
           .get_vector()),
      read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_1channel_2image_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape_a{2, 1, 14};
  Shape window_shape{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 1, 12};
  auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape),
                                 ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_plain_tensor(element::f32, shape_a);
  copy_data(
      a, test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                                  {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
             .get_vector());
  auto result = he_backend->create_plain_tensor(element::f32, shape_r);

  auto handle = backend->compile(f);
  handle->call_with_validate({result}, {a});
  EXPECT_TRUE(all_close(
      (test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}},
                                {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
           .get_vector()),
      read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_1channel_2image_plain_batched) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape_a{2, 1, 14};
  Shape window_shape{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 1, 12};
  auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape),
                                 ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_packed_plain_tensor(element::f32, shape_a);
  copy_data(
      a, test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                                  {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
             .get_vector());
  auto result = he_backend->create_packed_plain_tensor(element::f32, shape_r);

  auto handle = backend->compile(f);
  handle->call_with_validate({result}, {a});
  EXPECT_TRUE(all_close(
      (test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}},
                                {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
           .get_vector()),
      read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_1channel_2image_cipher) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape_a{2, 1, 14};
  Shape window_shape{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 1, 12};
  auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape),
                                 ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  copy_data(
      a, test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                                  {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
             .get_vector());
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  auto handle = backend->compile(f);
  handle->call_with_validate({result}, {a});
  EXPECT_TRUE(all_close(
      (test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}},
                                {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
           .get_vector()),
      read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_1channel_2image_cipher_batched) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape_a{2, 1, 14};
  Shape window_shape{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 1, 12};
  auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape),
                                 ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_packed_cipher_tensor(element::f32, shape_a);
  copy_data(
      a, test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                                  {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
             .get_vector());
  auto result = he_backend->create_packed_cipher_tensor(element::f32, shape_r);

  auto handle = backend->compile(f);
  handle->call_with_validate({result}, {a});
  EXPECT_TRUE(all_close(
      (test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}},
                                {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
           .get_vector()),
      read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_2channel_2image_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape_a{2, 2, 14};
  Shape window_shape{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 2, 12};
  auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape),
                                 ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_plain_tensor(element::f32, shape_a);
  copy_data(
      a, test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                   {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                                  {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                   {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})

             .get_vector());
  auto result = he_backend->create_plain_tensor(element::f32, shape_r);

  auto handle = backend->compile(f);
  handle->call_with_validate({result}, {a});
  EXPECT_TRUE(all_close(
      (test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0},
                                 {0, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1}},

                                {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2},
                                 {2, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2}}})
           .get_vector()),
      read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_2channel_2image_cipher) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape_a{2, 2, 14};
  Shape window_shape{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 2, 12};
  auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape),
                                 ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
  copy_data(
      a, test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                   {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                                  {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                   {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})

             .get_vector());
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  auto handle = backend->compile(f);
  handle->call_with_validate({result}, {a});
  EXPECT_TRUE(all_close(
      (test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0},
                                 {0, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1}},

                                {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2},
                                 {2, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2}}})
           .get_vector()),
      read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_2d_2channel_2image_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape_a{2, 2, 5, 5};
  Shape window_shape{2, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 2, 4, 3};
  auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape),
                                 ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_plain_tensor(element::f32, shape_a);
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
  auto result = he_backend->create_plain_tensor(element::f32, shape_r);

  auto handle = backend->compile(f);
  handle->call_with_validate({result}, {a});
  EXPECT_TRUE(all_close((test::NDArray<float, 4>({{{{3, 3, 2},  // img 0 chan 0
                                                    {3, 3, 2},
                                                    {2, 1, 2},
                                                    {2, 2, 2}},

                                                   {{3, 3, 3},  // img 0 chan 1
                                                    {3, 3, 3},
                                                    {3, 1, 2},
                                                    {3, 1, 0}}},

                                                  {{{2, 2, 2},  // img 1 chan 0
                                                    {2, 2, 3},
                                                    {2, 3, 3},
                                                    {2, 3, 3}},

                                                   {{2, 2, 1},  // img 1 chan 1
                                                    {2, 2, 2},
                                                    {2, 2, 2},
                                                    {1, 1, 2}}}})
                             .get_vector()),
                        read_vector<float>(result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_2d_2channel_2image_cipher) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape_a{2, 2, 5, 5};
  Shape window_shape{2, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 2, 4, 3};
  auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape),
                                 ParameterVector{A});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
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
  auto result = he_backend->create_cipher_tensor(element::f32, shape_r);

  auto handle = backend->compile(f);
  handle->call_with_validate({result}, {a});
  EXPECT_TRUE(all_close((test::NDArray<float, 4>({{{{3, 3, 2},  // img 0 chan 0
                                                    {3, 3, 2},
                                                    {2, 1, 2},
                                                    {2, 2, 2}},

                                                   {{3, 3, 3},  // img 0 chan 1
                                                    {3, 3, 3},
                                                    {3, 1, 2},
                                                    {3, 1, 0}}},

                                                  {{{2, 2, 2},  // img 1 chan 0
                                                    {2, 2, 3},
                                                    {2, 3, 3},
                                                    {2, 3, 3}},

                                                   {{2, 2, 1},  // img 1 chan 1
                                                    {2, 2, 2},
                                                    {2, 2, 2},
                                                    {1, 1, 2}}}})
                             .get_vector()),
                        read_vector<float>(result), 1e-3f));
}
