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

// Test multiplying cipher with cipher at different layer
NGRAPH_TEST(${BACKEND_NAME}, mult_layer_cipher_cipher) {
  Shape shape{2, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto C = make_shared<op::Parameter>(element::f32, shape);
  auto f = make_shared<Function>((A * B) * C, op::ParameterVector{A, B, C});

  auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(
      runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

  // Create some tensors for input/output
  auto a = backend->create_tensor(element::f32, shape);
  auto b = backend->create_tensor(element::f32, shape);
  auto c = backend->create_tensor(element::f32, shape);
  auto result = backend->create_tensor(element::f32, shape);

  copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
  copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  backend->call(f, {result}, {a, b, c});
  EXPECT_TRUE(all_close(
      read_vector<float>(result),
      (test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(), 1e-1f));

  backend->call(f, {result}, {b, a, c});
  EXPECT_TRUE(all_close(
      read_vector<float>(result),
      (test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(), 1e-1f));

  backend->call(f, {result}, {c, a, b});
  EXPECT_TRUE(all_close(
      read_vector<float>(result),
      (test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(), 1e-1f));
}

// Test multiplying cipher with plain at different layer
NGRAPH_TEST(${BACKEND_NAME}, mult_layer_cipher_plain) {
  Shape shape{2, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto C = make_shared<op::Parameter>(element::f32, shape);
  auto f = make_shared<Function>((A * B) * C, op::ParameterVector{A, B, C});

  auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(
      runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

  // Create some tensors for input/output
  auto a = backend->create_tensor(element::f32, shape);
  auto b = backend->create_tensor(element::f32, shape);
  auto c = backend->create_plain_tensor(element::f32, shape);
  auto result = backend->create_tensor(element::f32, shape);

  copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
  copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  backend->call(f, {result}, {a, b, c});
  EXPECT_TRUE(all_close(
      read_vector<float>(result),
      (test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(), 1e-1f));

  backend->call(f, {result}, {b, a, c});
  EXPECT_TRUE(all_close(
      read_vector<float>(result),
      (test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(), 1e-1f));

  backend->call(f, {result}, {c, a, b});
  EXPECT_TRUE(all_close(
      read_vector<float>(result),
      (test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(), 1e-1f));
}

// Test multiplying plain with plain at different layer
NGRAPH_TEST(${BACKEND_NAME}, mult_layer_plain_plain) {
  Shape shape{2, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto C = make_shared<op::Parameter>(element::f32, shape);
  auto f = make_shared<Function>((A * B) * C, op::ParameterVector{A, B, C});

  auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(
      runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

  // Create some tensors for input/output
  auto a = backend->create_plain_tensor(element::f32, shape);
  auto b = backend->create_plain_tensor(element::f32, shape);
  auto c = backend->create_plain_tensor(element::f32, shape);
  auto result = backend->create_tensor(element::f32, shape);

  copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
  copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  backend->call(f, {result}, {a, b, c});
  EXPECT_TRUE(all_close(
      read_vector<float>(result),
      (test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(), 1e-1f));

  backend->call(f, {result}, {b, a, c});
  EXPECT_TRUE(all_close(
      read_vector<float>(result),
      (test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(), 1e-1f));

  backend->call(f, {result}, {c, a, b});
  EXPECT_TRUE(all_close(
      read_vector<float>(result),
      (test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(), 1e-1f));
}

// Test adding cipher with cipher at different layer
NGRAPH_TEST(${BACKEND_NAME}, add_layer_cipher_cipher) {
  Shape shape{2, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto C = make_shared<op::Parameter>(element::f32, shape);
  auto f = make_shared<Function>((A * B) + C, op::ParameterVector{A, B, C});

  auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(
      runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

  // Create some tensors for input/output
  auto a = backend->create_tensor(element::f32, shape);
  auto b = backend->create_tensor(element::f32, shape);
  auto c = backend->create_tensor(element::f32, shape);
  auto result = backend->create_tensor(element::f32, shape);

  copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
  copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  backend->call(f, {result}, {a, b, c});
  EXPECT_TRUE(all_close(
      read_vector<float>(result),
      (test::NDArray<float, 2>({{14, 22}, {32, 44}})).get_vector(), 1e-1f));
}

// Test adding cipher with plain at different layer
NGRAPH_TEST(${BACKEND_NAME}, add_layer_cipher_plain) {
  Shape shape{2, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto C = make_shared<op::Parameter>(element::f32, shape);
  auto f = make_shared<Function>((A * B) + C, op::ParameterVector{A, B, C});

  auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(
      runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

  // Create some tensors for input/output
  auto a = backend->create_tensor(element::f32, shape);
  auto b = backend->create_tensor(element::f32, shape);
  auto c = backend->create_plain_tensor(element::f32, shape);
  auto result = backend->create_tensor(element::f32, shape);

  copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
  copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  backend->call(f, {result}, {a, b, c});
  EXPECT_TRUE(all_close(
      read_vector<float>(result),
      (test::NDArray<float, 2>({{14, 22}, {32, 44}})).get_vector(), 1e-1f));
}

// Test adding plain with plain at different layer
NGRAPH_TEST(${BACKEND_NAME}, add_layer_plain_plain) {
  Shape shape{2, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto C = make_shared<op::Parameter>(element::f32, shape);
  auto f = make_shared<Function>((A * B) + C, op::ParameterVector{A, B, C});

  auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(
      runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

  // Create some tensors for input/output
  auto a = backend->create_plain_tensor(element::f32, shape);
  auto b = backend->create_plain_tensor(element::f32, shape);
  auto c = backend->create_plain_tensor(element::f32, shape);
  auto result = backend->create_tensor(element::f32, shape);

  copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
  copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  backend->call(f, {result}, {a, b, c});
  EXPECT_TRUE(all_close(
      read_vector<float>(result),
      (test::NDArray<float, 2>({{14, 22}, {32, 44}})).get_vector(), 1e-1f));
}