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

static std::string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, minimum_plain) {
  ngraph::Shape shape{2, 2, 2};
  auto A = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto B = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto f = std::make_shared<ngraph::Function>(std::make_shared<ngraph::op::Minimum>(A, B),
                                 ngraph::ParameterVector{A, B});

  A->set_op_annotations(
      ngraph::he::HEOpAnnotations::server_plaintext_unpacked_annotation());
  B->set_op_annotations(
      ngraph::he::HEOpAnnotations::server_plaintext_unpacked_annotation());

  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  // Create some tensors for input/output
  auto a = he_backend->create_plain_tensor(ngraph::element::f32, shape);
  copy_data(a, std::vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
  auto b = he_backend->create_plain_tensor(ngraph::element::f32, shape);
  copy_data(b, std::vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
  auto result = he_backend->create_plain_tensor(ngraph::element::f32, shape);

  auto handle = he_backend->compile(f);
  handle->call_with_validate({result}, {a, b});
  EXPECT_TRUE(ngraph::test::he::all_close((std::vector<float>{1, 2, -8, 8, -0.5, 0, 1, 1}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, minimum_plain_packed) {
  ngraph::Shape shape{2, 2, 2};
  auto A = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto B = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto f = std::make_shared<ngraph::Function>(std::make_shared<ngraph::op::Minimum>(A, B),
                                 ngraph::ParameterVector{A, B});

  A->set_op_annotations(ngraph::he::HEOpAnnotations::server_plaintext_packed_annotation());
  B->set_op_annotations(ngraph::he::HEOpAnnotations::server_plaintext_packed_annotation());

  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  // Create some tensors for input/output
  auto a = he_backend->create_packed_plain_tensor(ngraph::element::f32, shape);
  copy_data(a, std::vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
  auto b = he_backend->create_packed_plain_tensor(ngraph::element::f32, shape);
  copy_data(b, std::vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
  auto result = he_backend->create_packed_plain_tensor(ngraph::element::f32, shape);

  auto handle = he_backend->compile(f);
  handle->call_with_validate({result}, {a, b});
  EXPECT_TRUE(ngraph::test::he::all_close((std::vector<float>{1, 2, -8, 8, -0.5, 0, 1, 1}),
                                  read_vector<float>(result)));
}
