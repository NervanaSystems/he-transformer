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

// Test multiplying cipher with cipher at different layer
NGRAPH_TEST(${BACKEND_NAME}, mult_layer_cipher_cipher) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  ngraph::Shape shape{2, 2};
  auto A = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto B = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto C = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto f = std::make_shared<ngraph::Function>((A * B) * C, ngraph::ParameterVector{A, B, C});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
  auto b = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
  auto c = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
  auto result = he_backend->create_cipher_tensor(ngraph::element::f32, shape);

  A->set_op_annotations(
      ngraph::he::HEOpAnnotations::server_ciphertext_unpacked_annotation());
  B->set_op_annotations(
      ngraph::he::HEOpAnnotations::server_ciphertext_unpacked_annotation());
  C->set_op_annotations(
      ngraph::he::HEOpAnnotations::server_ciphertext_unpacked_annotation());

  {
    copy_data(a, ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto handle1 = backend->compile(f);
    handle1->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(ngraph::test::he::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }

  {
    copy_data(a, ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto handle2 = backend->compile(f);
    handle2->call_with_validate({result}, {b, a, c});
    EXPECT_TRUE(ngraph::test::he::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }
  {
    copy_data(a, ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto handle3 = backend->compile(f);
    handle3->call_with_validate({result}, {c, a, b});
    EXPECT_TRUE(ngraph::test::he::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }
}

// Test multiplying cipher with plain at different layer
NGRAPH_TEST(${BACKEND_NAME}, mult_layer_cipher_plain) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  ngraph::Shape shape{2, 2};

  // A B C order
  {
    auto A = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
    auto B = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
    auto C = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
    auto f = std::make_shared<ngraph::Function>((A * B) * C, ngraph::ParameterVector{A, B, C});

    A->set_op_annotations(
        ngraph::he::HEOpAnnotations::server_ciphertext_unpacked_annotation());
    B->set_op_annotations(
        ngraph::he::HEOpAnnotations::server_ciphertext_unpacked_annotation());
    C->set_op_annotations(
        ngraph::he::HEOpAnnotations::server_plaintext_unpacked_annotation());

    auto a = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
    auto b = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
    auto c = he_backend->create_plain_tensor(ngraph::element::f32, shape);
    auto result = he_backend->create_cipher_tensor(ngraph::element::f32, shape);

    copy_data(a, ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(ngraph::test::he::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }
  // B A C order
  {
    auto A = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
    auto B = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
    auto C = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
    auto f = std::make_shared<ngraph::Function>((A * B) * C, ngraph::ParameterVector{A, B, C});

    A->set_op_annotations(
        ngraph::he::HEOpAnnotations::server_ciphertext_unpacked_annotation());
    B->set_op_annotations(
        ngraph::he::HEOpAnnotations::server_ciphertext_unpacked_annotation());
    C->set_op_annotations(
        ngraph::he::HEOpAnnotations::server_plaintext_unpacked_annotation());

    auto a = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
    auto b = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
    auto c = he_backend->create_plain_tensor(ngraph::element::f32, shape);
    auto result = he_backend->create_cipher_tensor(ngraph::element::f32, shape);

    copy_data(a, ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {b, a, c});
    EXPECT_TRUE(ngraph::test::he::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }
  // C A B order
  {
    auto A = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
    auto B = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
    auto C = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
    auto f = std::make_shared<ngraph::Function>((A * B) * C, ngraph::ParameterVector{A, B, C});

    A->set_op_annotations(
        ngraph::he::HEOpAnnotations::server_plaintext_unpacked_annotation());
    B->set_op_annotations(
        ngraph::he::HEOpAnnotations::server_ciphertext_unpacked_annotation());
    C->set_op_annotations(
        ngraph::he::HEOpAnnotations::server_ciphertext_unpacked_annotation());

    auto a = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
    auto b = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
    auto c = he_backend->create_plain_tensor(ngraph::element::f32, shape);
    auto result = he_backend->create_cipher_tensor(ngraph::element::f32, shape);

    copy_data(a, ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {c, a, b});
    EXPECT_TRUE(ngraph::test::he::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }
}

// Test multiplying plain with plain at different layer
NGRAPH_TEST(${BACKEND_NAME}, mult_layer_plain_plain) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  ngraph::Shape shape{2, 2};
  auto A = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto B = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto C = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto f = std::make_shared<ngraph::Function>((A * B) * C, ngraph::ParameterVector{A, B, C});

  // Create some tensors for input/output
  auto a = he_backend->create_plain_tensor(ngraph::element::f32, shape);
  auto b = he_backend->create_plain_tensor(ngraph::element::f32, shape);
  auto c = he_backend->create_plain_tensor(ngraph::element::f32, shape);
  auto result = he_backend->create_plain_tensor(ngraph::element::f32, shape);

  copy_data(a, ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
  copy_data(b, ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(c, ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  {
    auto handle1 = backend->compile(f);
    handle1->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(ngraph::test::he::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }
  {
    auto handle2 = backend->compile(f);
    handle2->call_with_validate({result}, {b, a, c});
    EXPECT_TRUE(ngraph::test::he::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }
  {
    auto handle3 = backend->compile(f);
    handle3->call_with_validate({result}, {c, a, b});
    EXPECT_TRUE(ngraph::test::he::all_close(
        read_vector<float>(result),
        (ngraph::test::NDArray<float, 2>({{45, 120}, {231, 384}})).get_vector(),
        1e-1f));
  }
}

// Test adding cipher with cipher at different layer
NGRAPH_TEST(${BACKEND_NAME}, add_layer_cipher_cipher) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  ngraph::Shape shape{2, 2};
  auto A = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto B = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto C = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto f = std::make_shared<ngraph::Function>((A * B) + C, ngraph::ParameterVector{A, B, C});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
  auto b = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
  auto c = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
  auto result = he_backend->create_cipher_tensor(ngraph::element::f32, shape);

  A->set_op_annotations(
      ngraph::he::HEOpAnnotations::server_ciphertext_unpacked_annotation());
  B->set_op_annotations(
      ngraph::he::HEOpAnnotations::server_ciphertext_unpacked_annotation());
  C->set_op_annotations(
      ngraph::he::HEOpAnnotations::server_ciphertext_unpacked_annotation());

  copy_data(a, ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
  copy_data(b, ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(c, ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  auto handle = backend->compile(f);
  handle->call_with_validate({result}, {a, b, c});
  EXPECT_TRUE(ngraph::test::he::all_close(
      read_vector<float>(result),
      (ngraph::test::NDArray<float, 2>({{14, 22}, {32, 44}})).get_vector(), 1e-1f));
}

// Test adding cipher with plain at different layer
NGRAPH_TEST(${BACKEND_NAME}, add_layer_cipher_plain) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  ngraph::Shape shape{2, 2};
  auto A = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto B = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto C = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto f = std::make_shared<ngraph::Function>((A * B) + C, ngraph::ParameterVector{A, B, C});

  // Create some tensors for input/output
  auto a = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
  auto b = he_backend->create_cipher_tensor(ngraph::element::f32, shape);
  auto c = he_backend->create_plain_tensor(ngraph::element::f32, shape);
  auto result = he_backend->create_cipher_tensor(ngraph::element::f32, shape);

  A->set_op_annotations(
      ngraph::he::HEOpAnnotations::server_ciphertext_unpacked_annotation());
  B->set_op_annotations(
      ngraph::he::HEOpAnnotations::server_ciphertext_unpacked_annotation());
  C->set_op_annotations(
      ngraph::he::HEOpAnnotations::server_plaintext_unpacked_annotation());

  copy_data(a, ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
  copy_data(b, ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(c, ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  auto handle = backend->compile(f);
  handle->call_with_validate({result}, {a, b, c});
  EXPECT_TRUE(ngraph::test::he::all_close(
      read_vector<float>(result),
      (ngraph::test::NDArray<float, 2>({{14, 22}, {32, 44}})).get_vector(), 1e-1f));
}

// Test adding plain with plain at different layer
NGRAPH_TEST(${BACKEND_NAME}, add_layer_plain_plain) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  ngraph::Shape shape{2, 2};
  auto A = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto B = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto C = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto f = std::make_shared<ngraph::Function>((A * B) + C, ngraph::ParameterVector{A, B, C});

  // Create some tensors for input/output
  auto a = he_backend->create_plain_tensor(ngraph::element::f32, shape);
  auto b = he_backend->create_plain_tensor(ngraph::element::f32, shape);
  auto c = he_backend->create_plain_tensor(ngraph::element::f32, shape);
  auto result = he_backend->create_plain_tensor(ngraph::element::f32, shape);

  copy_data(a, ngraph::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
  copy_data(b, ngraph::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
  copy_data(c, ngraph::test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

  auto handle = backend->compile(f);
  handle->call_with_validate({result}, {a, b, c});
  EXPECT_TRUE(ngraph::test::he::all_close(
      read_vector<float>(result),
      (ngraph::test::NDArray<float, 2>({{14, 22}, {32, 44}})).get_vector(), 1e-1f));
}
