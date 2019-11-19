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
#include "ngraph/pass/visualize_tree.hpp"
#include "seal/he_seal_backend.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

static std::string s_manifest = "${MANIFEST}";

namespace ngraph::runtime::he {

// Test multiplying cipher with cipher at different layer
NGRAPH_TEST(${BACKEND_NAME}, mult_layer_cipher_cipher) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{2, 2};
  auto a = std::make_shared<op::Parameter>(element::f32, shape);
  auto b = std::make_shared<op::Parameter>(element::f32, shape);
  auto c = std::make_shared<op::Parameter>(element::f32, shape);
  auto f = std::make_shared<Function>((a * b) * c, ParameterVector{a, b, c});

  // Create some tensors for input/output
  auto t_a = he_backend->create_cipher_tensor(element::f32, shape);
  auto t_b = he_backend->create_cipher_tensor(element::f32, shape);
  auto t_c = he_backend->create_cipher_tensor(element::f32, shape);
  auto result = he_backend->create_cipher_tensor(element::f32, shape);

  auto cipher_annotation =
      HEOpAnnotations::server_ciphertext_unpacked_annotation();
  const auto& cipher_config = test::config_from_annotation(*cipher_annotation);
  std::string error_str;
  he_backend->set_config({{a->get_name(), cipher_config},
                          {b->get_name(), cipher_config},
                          {c->get_name(), cipher_config}},
                         error_str);

  {
    copy_data(t_a,
              ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(t_b,
              ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(
        t_c, ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto handle1 = backend->compile(f);
    handle1->call_with_validate({result}, {t_a, t_b, t_c});
    EXPECT_TRUE(test::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }

  {
    copy_data(t_a,
              ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(t_b,
              ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(
        t_c, ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto handle2 = backend->compile(f);
    handle2->call_with_validate({result}, {t_b, t_a, t_c});
    EXPECT_TRUE(test::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }
  {
    copy_data(t_a,
              ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(t_b,
              ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(
        t_c, ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto handle3 = backend->compile(f);
    handle3->call_with_validate({result}, {t_c, t_a, t_b});
    EXPECT_TRUE(test::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }
}

// Test multiplying cipher with plain at different layer
NGRAPH_TEST(${BACKEND_NAME}, mult_layer_cipher_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{2, 2};

  auto cipher_annotation =
      HEOpAnnotations::server_ciphertext_unpacked_annotation();
  auto plain_annotation =
      HEOpAnnotations::server_plaintext_unpacked_annotation();

  const auto& cipher_config = test::config_from_annotation(*cipher_annotation);
  const auto& plain_config = test::config_from_annotation(*plain_annotation);

  std::string error_str;

  // A B C order
  {
    auto a = std::make_shared<op::Parameter>(element::f32, shape);
    auto b = std::make_shared<op::Parameter>(element::f32, shape);
    auto c = std::make_shared<op::Parameter>(element::f32, shape);
    auto f = std::make_shared<Function>((a * b) * c, ParameterVector{a, b, c});

    he_backend->set_config({{a->get_name(), cipher_config},
                            {b->get_name(), cipher_config},
                            {c->get_name(), plain_config}},
                           error_str);

    auto t_a = he_backend->create_cipher_tensor(element::f32, shape);
    auto t_b = he_backend->create_cipher_tensor(element::f32, shape);
    auto t_c = he_backend->create_plain_tensor(element::f32, shape);
    auto result = he_backend->create_cipher_tensor(element::f32, shape);

    copy_data(t_a,
              ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(t_b,
              ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(
        t_c, ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {t_a, t_b, t_c});
    EXPECT_TRUE(test::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }
  // B A C order
  {
    auto a = std::make_shared<op::Parameter>(element::f32, shape);
    auto b = std::make_shared<op::Parameter>(element::f32, shape);
    auto c = std::make_shared<op::Parameter>(element::f32, shape);
    auto f = std::make_shared<Function>((a * b) * c, ParameterVector{a, b, c});

    he_backend->set_config({{a->get_name(), cipher_config},
                            {b->get_name(), cipher_config},
                            {c->get_name(), plain_config}},
                           error_str);

    auto t_a = he_backend->create_cipher_tensor(element::f32, shape);
    auto t_b = he_backend->create_cipher_tensor(element::f32, shape);
    auto t_c = he_backend->create_plain_tensor(element::f32, shape);
    auto result = he_backend->create_cipher_tensor(element::f32, shape);

    copy_data(t_a,
              ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(t_b,
              ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(
        t_c, ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {t_b, t_a, t_c});
    EXPECT_TRUE(test::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }
  // C A B order
  {
    auto a = std::make_shared<op::Parameter>(element::f32, shape);
    auto b = std::make_shared<op::Parameter>(element::f32, shape);
    auto c = std::make_shared<op::Parameter>(element::f32, shape);
    auto f = std::make_shared<Function>((a * b) * c, ParameterVector{a, b, c});

    he_backend->set_config({{a->get_name(), plain_config},
                            {b->get_name(), cipher_config},
                            {c->get_name(), cipher_config}},
                           error_str);

    auto t_a = he_backend->create_cipher_tensor(element::f32, shape);
    auto t_b = he_backend->create_cipher_tensor(element::f32, shape);
    auto t_c = he_backend->create_plain_tensor(element::f32, shape);
    auto result = he_backend->create_cipher_tensor(element::f32, shape);

    copy_data(t_a,
              ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(t_b,
              ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(
        t_c, ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {t_c, t_a, t_b});
    EXPECT_TRUE(test::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }
}

// Test multiplying plain with plain at different layer
NGRAPH_TEST(${BACKEND_NAME}, mult_layer_plain_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{2, 2};
  auto a = std::make_shared<op::Parameter>(element::f32, shape);
  auto b = std::make_shared<op::Parameter>(element::f32, shape);
  auto c = std::make_shared<op::Parameter>(element::f32, shape);
  auto f = std::make_shared<Function>((a * b) * c, ParameterVector{a, b, c});

  // Create some tensors for input/output
  auto t_a = he_backend->create_plain_tensor(element::f32, shape);
  auto t_b = he_backend->create_plain_tensor(element::f32, shape);
  auto t_c = he_backend->create_plain_tensor(element::f32, shape);
  auto result = he_backend->create_plain_tensor(element::f32, shape);

  copy_data(t_a,
            ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
  copy_data(t_b,
            ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(t_c,
            ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  {
    auto handle1 = backend->compile(f);
    handle1->call_with_validate({result}, {t_a, t_b, t_c});
    EXPECT_TRUE(test::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }
  {
    auto handle2 = backend->compile(f);
    handle2->call_with_validate({result}, {t_b, t_a, t_c});
    EXPECT_TRUE(test::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }
  {
    auto handle3 = backend->compile(f);
    handle3->call_with_validate({result}, {t_c, t_a, t_b});
    EXPECT_TRUE(test::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }
}

// Test adding cipher with cipher at different layer
NGRAPH_TEST(${BACKEND_NAME}, add_layer_cipher_cipher) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{2, 2};
  auto a = std::make_shared<op::Parameter>(element::f32, shape);
  auto b = std::make_shared<op::Parameter>(element::f32, shape);
  auto c = std::make_shared<op::Parameter>(element::f32, shape);
  auto f = std::make_shared<Function>((a * b) + c, ParameterVector{a, b, c});

  // Create some tensors for input/output
  auto t_a = he_backend->create_cipher_tensor(element::f32, shape);
  auto t_b = he_backend->create_cipher_tensor(element::f32, shape);
  auto t_c = he_backend->create_cipher_tensor(element::f32, shape);
  auto result = he_backend->create_cipher_tensor(element::f32, shape);

  auto cipher_annotation =
      HEOpAnnotations::server_ciphertext_unpacked_annotation();
  const auto& cipher_config = test::config_from_annotation(*cipher_annotation);

  std::string error_str;
  he_backend->set_config({{a->get_name(), cipher_config},
                          {b->get_name(), cipher_config},
                          {c->get_name(), cipher_config}},
                         error_str);

  copy_data(t_a,
            ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
  copy_data(t_b,
            ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(t_c,
            ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  auto handle = backend->compile(f);
  handle->call_with_validate({result}, {t_a, t_b, t_c});
  EXPECT_TRUE(test::all_close(
      read_vector<float>(result),
      (ngraph::test::NDArray<float, 2>({{14, 22}, {32, 44}})).get_vector(),
      1e-1f));
}

// Test adding cipher with plain at different layer
NGRAPH_TEST(${BACKEND_NAME}, add_layer_cipher_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{2, 2};
  auto a = std::make_shared<op::Parameter>(element::f32, shape);
  auto b = std::make_shared<op::Parameter>(element::f32, shape);
  auto c = std::make_shared<op::Parameter>(element::f32, shape);
  auto f = std::make_shared<Function>((a * b) + c, ParameterVector{a, b, c});

  // Create some tensors for input/output
  auto t_a = he_backend->create_cipher_tensor(element::f32, shape);
  auto t_b = he_backend->create_cipher_tensor(element::f32, shape);
  auto t_c = he_backend->create_plain_tensor(element::f32, shape);
  auto result = he_backend->create_cipher_tensor(element::f32, shape);

  auto cipher_annotation =
      HEOpAnnotations::server_ciphertext_unpacked_annotation();
  auto plain_annotation =
      HEOpAnnotations::server_plaintext_unpacked_annotation();

  const auto& cipher_config = test::config_from_annotation(*cipher_annotation);
  const auto& plain_config = test::config_from_annotation(*plain_annotation);

  std::string error_str;
  he_backend->set_config({{a->get_name(), cipher_config},
                          {b->get_name(), cipher_config},
                          {c->get_name(), plain_config}},
                         error_str);

  copy_data(t_a,
            ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
  copy_data(t_b,
            ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(t_c,
            ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  auto handle = backend->compile(f);
  handle->call_with_validate({result}, {t_a, t_b, t_c});
  EXPECT_TRUE(test::all_close(
      read_vector<float>(result),
      (ngraph::test::NDArray<float, 2>({{14, 22}, {32, 44}})).get_vector(),
      1e-1f));
}

// Test adding plain with plain at different layer
NGRAPH_TEST(${BACKEND_NAME}, add_layer_plain_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{2, 2};
  auto a = std::make_shared<op::Parameter>(element::f32, shape);
  auto b = std::make_shared<op::Parameter>(element::f32, shape);
  auto c = std::make_shared<op::Parameter>(element::f32, shape);
  auto f = std::make_shared<Function>((a * b) + c, ParameterVector{a, b, c});

  // Create some tensors for input/output
  auto t_a = he_backend->create_plain_tensor(element::f32, shape);
  auto t_b = he_backend->create_plain_tensor(element::f32, shape);
  auto t_c = he_backend->create_plain_tensor(element::f32, shape);
  auto result = he_backend->create_plain_tensor(element::f32, shape);

  copy_data(t_a,
            ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
  copy_data(t_b,
            ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(t_c,
            ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  auto handle = backend->compile(f);
  handle->call_with_validate({result}, {t_a, t_b, t_c});
  EXPECT_TRUE(test::all_close(
      read_vector<float>(result),
      (ngraph::test::NDArray<float, 2>({{14, 22}, {32, 44}})).get_vector(),
      1e-1f));
}

}  // namespace ngraph::runtime::he
