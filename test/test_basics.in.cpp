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

NGRAPH_TEST(${BACKEND_NAME}, trivial) {
  int x = 1;
  int y = 2;
  int z = x + y;
  EXPECT_EQ(z, 3);
}

NGRAPH_TEST(${BACKEND_NAME}, create_backend) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, create_tensor) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape{2, 2};
  backend->create_tensor(element::f32, shape);
}

NGRAPH_TEST(${BACKEND_NAME}, create_cipher_tensor) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape{2, 2};
  he_backend->create_cipher_tensor(element::f32, shape);
}

NGRAPH_TEST(${BACKEND_NAME}, create_plain_tensor) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape{2, 2};
  he_backend->create_plain_tensor(element::f32, shape);
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_count) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape{2, 2};

  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto f =
      make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

  auto a = backend->create_tensor(element::f32, shape);
  auto b = backend->create_tensor(element::f32, shape);
  auto c = backend->create_tensor(element::f32, shape);

  auto handle = backend->compile(f);

  EXPECT_ANY_THROW(handle->call_with_validate({c}, {a}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_type) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape{2, 2};

  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto f =
      make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

  auto a = backend->create_tensor(element::i32, shape);
  auto b = backend->create_tensor(element::f32, shape);
  auto c = backend->create_tensor(element::f32, shape);

  auto handle = backend->compile(f);

  EXPECT_ANY_THROW(handle->call_with_validate({c}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_shape) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape{2, 2};

  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto f =
      make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

  auto a = backend->create_tensor(element::f32, {2, 3});
  auto b = backend->create_tensor(element::f32, shape);
  auto c = backend->create_tensor(element::f32, shape);

  auto handle = backend->compile(f);

  EXPECT_ANY_THROW(handle->call_with_validate({c}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_count) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape{2, 2};

  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto f =
      make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

  auto a = backend->create_tensor(element::f32, shape);
  auto b = backend->create_tensor(element::f32, shape);
  auto c = backend->create_tensor(element::f32, shape);
  auto d = backend->create_tensor(element::f32, shape);

  auto handle = backend->compile(f);

  EXPECT_ANY_THROW(handle->call_with_validate({c, d}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_type) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape{2, 2};

  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto f =
      make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

  auto a = backend->create_tensor(element::i32, shape);
  auto b = backend->create_tensor(element::f32, shape);
  auto c = backend->create_tensor(element::f32, shape);

  auto handle = backend->compile(f);

  EXPECT_ANY_THROW(handle->call_with_validate({a}, {b, c}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_shape) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape{2, 2};

  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto f =
      make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

  auto a = backend->create_tensor(element::f32, {2, 3});
  auto b = backend->create_tensor(element::f32, shape);
  auto c = backend->create_tensor(element::f32, shape);

  auto handle = backend->compile(f);

  EXPECT_ANY_THROW(handle->call_with_validate({a}, {c, b}));
}
