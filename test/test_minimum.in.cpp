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

NGRAPH_TEST(${BACKEND_NAME}, minimum_plain) {
  Shape shape{2, 2, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto f = make_shared<Function>(make_shared<op::Minimum>(A, B),
                                 ParameterVector{A, B});

  A->set_op_annotations(
      HEOpAnnotations::server_plaintext_unpacked_annotation());
  B->set_op_annotations(
      HEOpAnnotations::server_plaintext_unpacked_annotation());

  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  // Create some tensors for input/output
  auto a = he_backend->create_plain_tensor(element::f32, shape);
  copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
  auto b = he_backend->create_plain_tensor(element::f32, shape);
  copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
  auto result = he_backend->create_plain_tensor(element::f32, shape);

  auto handle = he_backend->compile(f);
  handle->call_with_validate({result}, {a, b});
  EXPECT_TRUE(all_close((vector<float>{1, 2, -8, 8, -.5, 0, 1, 1}),
                        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, minimum_plain_packed) {
  Shape shape{2, 2, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto f = make_shared<Function>(make_shared<op::Minimum>(A, B),
                                 ParameterVector{A, B});

  A->set_op_annotations(HEOpAnnotations::server_plaintext_packed_annotation());
  B->set_op_annotations(HEOpAnnotations::server_plaintext_packed_annotation());

  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  // Create some tensors for input/output
  auto a = he_backend->create_packed_plain_tensor(element::f32, shape);
  copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
  auto b = he_backend->create_packed_plain_tensor(element::f32, shape);
  copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
  auto result = he_backend->create_packed_plain_tensor(element::f32, shape);

  auto handle = he_backend->compile(f);
  handle->call_with_validate({result}, {a, b});
  EXPECT_TRUE(all_close((vector<float>{1, 2, -8, 8, -.5, 0, 1, 1}),
                        read_vector<float>(result)));
}
