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

#include "he_op_annotations.hpp"
#include "ngraph/ngraph.hpp"
#include "seal/he_seal_backend.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::he;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, constant) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape{2, 2};
  {
    auto A = op::Constant::create(element::f32, shape, {0.1, 0.2, 0.3, 0.4});
    auto f = make_shared<Function>(A, ParameterVector{});
    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {});
    EXPECT_TRUE(all_close((vector<float>{0.1, 0.2, 0.3, 0.4}),
                          read_vector<float>(result)));
  }
  {
    auto A = op::Constant::create(element::f64, shape, {0.1, 0.2, 0.3, 0.4});
    auto f = make_shared<Function>(A, ParameterVector{});
    auto result = backend->create_tensor(element::f64, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {});
    EXPECT_TRUE(all_close((vector<double>{0.1, 0.2, 0.3, 0.4}),
                          read_vector<double>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, constant_abc_plain_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape{2, 2};
  auto a = op::Constant::create(element::f32, shape, {1, 2, 3, 4});
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto c = make_shared<op::Parameter>(element::f32, shape);
  auto t = (a + b) * c;
  auto f = make_shared<Function>(t, ParameterVector{b, c});

  auto t_b = he_backend->create_plain_tensor(element::f32, shape);
  auto t_c = he_backend->create_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);

  copy_data(t_b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(t_c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_b, t_c});
  EXPECT_TRUE(all_close(
      read_vector<float>(t_result),
      (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector()));
}

NGRAPH_TEST(${BACKEND_NAME}, constant_abc_plain_cipher) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape{2, 2};
  auto a = op::Constant::create(element::f32, shape, {1, 2, 3, 4});
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto c = make_shared<op::Parameter>(element::f32, shape);
  auto t = (a + b) * c;
  auto f = make_shared<Function>(t, ParameterVector{b, c});

  b->set_op_annotations(
      HEOpAnnotations::server_plaintext_unpacked_annotation());
  c->set_op_annotations(
      HEOpAnnotations::server_ciphertext_unpacked_annotation());

  auto t_b = he_backend->create_plain_tensor(element::f32, shape);
  auto t_c = he_backend->create_cipher_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);

  copy_data(t_b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(t_c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_b, t_c});
  EXPECT_TRUE(all_close(
      read_vector<float>(t_result),
      (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector()));
}

NGRAPH_TEST(${BACKEND_NAME}, constant_abc_cipher_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape{2, 2};
  auto a = op::Constant::create(element::f32, shape, {1, 2, 3, 4});
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto c = make_shared<op::Parameter>(element::f32, shape);
  auto t = (a + b) * c;
  auto f = make_shared<Function>(t, ParameterVector{b, c});

  b->set_op_annotations(
      HEOpAnnotations::server_ciphertext_unpacked_annotation());
  c->set_op_annotations(
      HEOpAnnotations::server_plaintext_unpacked_annotation());

  auto t_b = he_backend->create_cipher_tensor(element::f32, shape);
  auto t_c = he_backend->create_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);

  copy_data(t_b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(t_c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_b, t_c});
  EXPECT_TRUE(all_close(
      read_vector<float>(t_result),
      (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector()));
}

NGRAPH_TEST(${BACKEND_NAME}, constant_abc_cipher_cipher) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape{2, 2};
  auto a = op::Constant::create(element::f32, shape, {1, 2, 3, 4});
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto c = make_shared<op::Parameter>(element::f32, shape);
  auto t = (a + b) * c;
  auto f = make_shared<Function>(t, ParameterVector{b, c});

  b->set_op_annotations(
      HEOpAnnotations::server_ciphertext_unpacked_annotation());
  c->set_op_annotations(
      HEOpAnnotations::server_ciphertext_unpacked_annotation());

  auto t_b = he_backend->create_cipher_tensor(element::f32, shape);
  auto t_c = he_backend->create_cipher_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);

  copy_data(t_b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(t_c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_b, t_c});
  EXPECT_TRUE(all_close(
      read_vector<float>(t_result),
      (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector()));
}
