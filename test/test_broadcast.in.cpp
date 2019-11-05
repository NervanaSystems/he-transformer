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

auto broadcast_test = [](const ngraph::Shape& shape_a,
                         const ngraph::Shape& shape_r,
                         const ngraph::AxisSet& axis_set,
                         const std::vector<float>& input,
                         const std::vector<float>& output,
                         const bool arg1_encrypted, const bool complex_packing,
                         const bool packed) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        ngraph::he::HESealEncryptionParameters::
            default_complex_packing_parms());
  }

  auto a =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape_a);
  auto t = std::make_shared<ngraph::op::Broadcast>(a, shape_r, axis_set);
  auto f = std::make_shared<ngraph::Function>(t, ngraph::ParameterVector{a});

  a->set_op_annotations(
      ngraph::test::he::annotation_from_flags(false, arg1_encrypted, packed));

  auto t_a = ngraph::test::he::tensor_from_flags(*he_backend, shape_a,
                                                 arg1_encrypted, packed);
  auto t_result = ngraph::test::he::tensor_from_flags(*he_backend, shape_r,
                                                      arg1_encrypted, packed);

  copy_data(t_a, input);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a});
  EXPECT_TRUE(
      ngraph::test::he::all_close(read_vector<float>(t_result), output, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_vector_plain_real_unpacked) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{4}, ngraph::AxisSet{0},
                 std::vector<float>{6}, std::vector<float>{6, 6, 6, 6}, false,
                 false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_vector_plain_real_packed) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{4}, ngraph::AxisSet{0},
                 std::vector<float>{6}, std::vector<float>{6, 6, 6, 6}, false,
                 false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_vector_plain_complex_unpacked) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{4}, ngraph::AxisSet{0},
                 std::vector<float>{6}, std::vector<float>{6, 6, 6, 6}, false,
                 true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_vector_plain_complex_packed) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{4}, ngraph::AxisSet{0},
                 std::vector<float>{6}, std::vector<float>{6, 6, 6, 6}, false,
                 true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_vector_cipher_real_unpacked) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{4}, ngraph::AxisSet{0},
                 std::vector<float>{6}, std::vector<float>{6, 6, 6, 6}, true,
                 false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_vector_cipher_real_packed) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{4}, ngraph::AxisSet{0},
                 std::vector<float>{6}, std::vector<float>{6, 6, 6, 6}, true,
                 false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_vector_cipher_complex_unpacked) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{4}, ngraph::AxisSet{0},
                 std::vector<float>{6}, std::vector<float>{6, 6, 6, 6}, true,
                 true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_vector_cipher_complex_packed) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{4}, ngraph::AxisSet{0},
                 std::vector<float>{6}, std::vector<float>{6, 6, 6, 6}, true,
                 true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_matrix_plain_real_unpacked) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{2, 2}, ngraph::AxisSet{0, 1},
                 std::vector<float>{6}, std::vector<float>{6, 6, 6, 6}, false,
                 false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_matrix_plain_real_packed) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{2, 2}, ngraph::AxisSet{0, 1},
                 std::vector<float>{6}, std::vector<float>{6, 6, 6, 6}, false,
                 false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_matrix_plain_complex_unpacked) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{2, 2}, ngraph::AxisSet{0, 1},
                 std::vector<float>{6}, std::vector<float>{6, 6, 6, 6}, false,
                 true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_matrix_plain_complex_packed) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{2, 2}, ngraph::AxisSet{0, 1},
                 std::vector<float>{6}, std::vector<float>{6, 6, 6, 6}, false,
                 true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_matrix_cipher_real_unpacked) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{2, 2}, ngraph::AxisSet{0, 1},
                 std::vector<float>{6}, std::vector<float>{6, 6, 6, 6}, true,
                 false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_matrix_cipher_real_packed) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{2, 2}, ngraph::AxisSet{0, 1},
                 std::vector<float>{6}, std::vector<float>{6, 6, 6, 6}, true,
                 false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_matrix_cipher_complex_unpacked) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{2, 2}, ngraph::AxisSet{0, 1},
                 std::vector<float>{6}, std::vector<float>{6, 6, 6, 6}, true,
                 true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_matrix_cipher_complex_packed) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{2, 2}, ngraph::AxisSet{0, 1},
                 std::vector<float>{6}, std::vector<float>{6, 6, 6, 6}, true,
                 true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_tensor_plain_real_unpacked) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{2, 2, 2},
                 ngraph::AxisSet{0, 1, 2}, std::vector<float>{6},
                 std::vector<float>{6, 6, 6, 6, 6, 6, 6, 6}, false, false,
                 false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_tensor_plain_real_packed) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{2, 2, 2},
                 ngraph::AxisSet{0, 1, 2}, std::vector<float>{6},
                 std::vector<float>{6, 6, 6, 6, 6, 6, 6, 6}, false, false,
                 true);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_tensor_plain_complex_unpacked) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{2, 2, 2},
                 ngraph::AxisSet{0, 1, 2}, std::vector<float>{6},
                 std::vector<float>{6, 6, 6, 6, 6, 6, 6, 6}, false, true,
                 false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_tensor_plain_complex_packed) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{2, 2, 2},
                 ngraph::AxisSet{0, 1, 2}, std::vector<float>{6},
                 std::vector<float>{6, 6, 6, 6, 6, 6, 6, 6}, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_tensor_cipher_real_unpacked) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{2, 2, 2},
                 ngraph::AxisSet{0, 1, 2}, std::vector<float>{6},
                 std::vector<float>{6, 6, 6, 6, 6, 6, 6, 6}, true, false,
                 false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_tensor_cipher_real_packed) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{2, 2, 2},
                 ngraph::AxisSet{0, 1, 2}, std::vector<float>{6},
                 std::vector<float>{6, 6, 6, 6, 6, 6, 6, 6}, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_tensor_cipher_complex_unpacked) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{2, 2, 2},
                 ngraph::AxisSet{0, 1, 2}, std::vector<float>{6},
                 std::vector<float>{6, 6, 6, 6, 6, 6, 6, 6}, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_tensor_cipher_complex_packed) {
  broadcast_test(ngraph::Shape{}, ngraph::Shape{2, 2, 2},
                 ngraph::AxisSet{0, 1, 2}, std::vector<float>{6},
                 std::vector<float>{6, 6, 6, 6, 6, 6, 6, 6}, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_trivial_plain_real_unpacked) {
  broadcast_test(
      ngraph::Shape{2, 2, 2}, ngraph::Shape{2, 2, 2}, ngraph::AxisSet{},
      std::vector<float>{2, 4, 6, 8, 16, 32, 64, 128},
      std::vector<float>{2, 4, 6, 8, 16, 32, 64, 128}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_vector_colwise_plain_real_unpacked) {
  broadcast_test(ngraph::Shape{3}, ngraph::Shape{3, 4}, ngraph::AxisSet{1},
                 std::vector<float>{1, 2, 3},
                 std::vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}, false,
                 false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_vector_rowwise_plain_real_unpacked) {
  broadcast_test(ngraph::Shape{4}, ngraph::Shape{3, 4}, ngraph::AxisSet{0},
                 std::vector<float>{1, 2, 3, 4},
                 std::vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}, false,
                 false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_matrix_0_plain_real_unpacked) {
  broadcast_test(ngraph::Shape{2, 2}, ngraph::Shape{2, 2, 2},
                 ngraph::AxisSet{0}, std::vector<float>{1, 2, 3, 4},
                 std::vector<float>{1, 2, 3, 4, 1, 2, 3, 4}, false, false,
                 false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_matrix_1_plain_real_unpacked) {
  broadcast_test(ngraph::Shape{2, 2}, ngraph::Shape{2, 2, 2},
                 ngraph::AxisSet{1}, std::vector<float>{1, 2, 3, 4},
                 std::vector<float>{1, 2, 1, 2, 3, 4, 3, 4}, false, false,
                 false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_matrix_2_plain_real_unpacked) {
  broadcast_test(ngraph::Shape{2, 2}, ngraph::Shape{2, 2, 2},
                 ngraph::AxisSet{2}, std::vector<float>{1, 2, 3, 4},
                 std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4}, false, false,
                 false);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_to_non_existent_axis) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  ngraph::Shape shape_a{};
  auto a =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape_a);
  ngraph::Shape shape_r{4};
  ASSERT_THROW(auto f = std::make_shared<ngraph::Function>(
                   std::make_shared<ngraph::op::Broadcast>(
                       a, shape_r, ngraph::AxisSet{0, 1}),
                   ngraph::ParameterVector{a}),
               ngraph::ngraph_error);
}
