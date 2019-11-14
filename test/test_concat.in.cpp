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

auto concat_test = [](const ngraph::Shape& shape_a,
                      const ngraph::Shape& shape_b,
                      const ngraph::Shape& shape_c, size_t concat_axis,
                      const std::vector<float>& input_a,
                      const std::vector<float>& input_b,
                      const std::vector<float>& input_c,
                      const std::vector<float>& output,
                      const bool args_encrypted, const bool complex_packing,
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
  auto b =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape_b);
  auto c =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape_c);
  auto t = std::make_shared<ngraph::op::Concat>(ngraph::NodeVector{a, b, c},
                                                concat_axis);
  auto f =
      std::make_shared<ngraph::Function>(t, ngraph::ParameterVector{a, b, c});

  const auto& a_config =
      ngraph::test::he::config_from_flags(false, args_encrypted, packed);
  const auto& b_config =
      ngraph::test::he::config_from_flags(false, args_encrypted, packed);
  const auto& c_config =
      ngraph::test::he::config_from_flags(false, args_encrypted, packed);

  std::string error_str;
  he_backend->set_config({{a->get_name(), a_config},
                          {b->get_name(), b_config},
                          {c->get_name(), c_config}},
                         error_str);

  auto t_a = ngraph::test::he::tensor_from_flags(*he_backend, shape_a,
                                                 args_encrypted, packed);
  auto t_b = ngraph::test::he::tensor_from_flags(*he_backend, shape_b,
                                                 args_encrypted, packed);
  auto t_c = ngraph::test::he::tensor_from_flags(*he_backend, shape_c,
                                                 args_encrypted, packed);
  auto t_result = ngraph::test::he::tensor_from_flags(
      *he_backend, t->get_shape(), args_encrypted, packed);

  copy_data(t_a, input_a);
  copy_data(t_b, input_b);
  copy_data(t_c, input_c);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a, t_b, t_c});
  EXPECT_TRUE(
      ngraph::test::he::all_close(read_vector<float>(t_result), output, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise_plain_real_unpacked) {
  concat_test(
      ngraph::Shape{2, 2}, ngraph::Shape{2, 3}, ngraph::Shape{2, 3}, 1,
      std::vector<float>{2, 4, 8, 16}, std::vector<float>{1, 2, 4, 8, 16, 32},
      std::vector<float>{2, 3, 5, 7, 11, 13},
      std::vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13},
      false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise_plain_real_packed) {
  concat_test(
      ngraph::Shape{2, 2}, ngraph::Shape{2, 3}, ngraph::Shape{2, 3}, 1,
      std::vector<float>{2, 4, 8, 16}, std::vector<float>{1, 2, 4, 8, 16, 32},
      std::vector<float>{2, 3, 5, 7, 11, 13},
      std::vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13},
      false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise_plain_complex_unpacked) {
  concat_test(
      ngraph::Shape{2, 2}, ngraph::Shape{2, 3}, ngraph::Shape{2, 3}, 1,
      std::vector<float>{2, 4, 8, 16}, std::vector<float>{1, 2, 4, 8, 16, 32},
      std::vector<float>{2, 3, 5, 7, 11, 13},
      std::vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13},
      false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise_plain_complex_packed) {
  concat_test(
      ngraph::Shape{2, 2}, ngraph::Shape{2, 3}, ngraph::Shape{2, 3}, 1,
      std::vector<float>{2, 4, 8, 16}, std::vector<float>{1, 2, 4, 8, 16, 32},
      std::vector<float>{2, 3, 5, 7, 11, 13},
      std::vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13},
      false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise_cipher_real_unpacked) {
  concat_test(
      ngraph::Shape{2, 2}, ngraph::Shape{2, 3}, ngraph::Shape{2, 3}, 1,
      std::vector<float>{2, 4, 8, 16}, std::vector<float>{1, 2, 4, 8, 16, 32},
      std::vector<float>{2, 3, 5, 7, 11, 13},
      std::vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13},
      true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise_cipher_real_packed) {
  concat_test(
      ngraph::Shape{2, 2}, ngraph::Shape{2, 3}, ngraph::Shape{2, 3}, 1,
      std::vector<float>{2, 4, 8, 16}, std::vector<float>{1, 2, 4, 8, 16, 32},
      std::vector<float>{2, 3, 5, 7, 11, 13},
      std::vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13},
      false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise_cipher_complex_unpacked) {
  concat_test(
      ngraph::Shape{2, 2}, ngraph::Shape{2, 3}, ngraph::Shape{2, 3}, 1,
      std::vector<float>{2, 4, 8, 16}, std::vector<float>{1, 2, 4, 8, 16, 32},
      std::vector<float>{2, 3, 5, 7, 11, 13},
      std::vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13},
      true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise_cipher_complex_packed) {
  concat_test(
      ngraph::Shape{2, 2}, ngraph::Shape{2, 3}, ngraph::Shape{2, 3}, 1,
      std::vector<float>{2, 4, 8, 16}, std::vector<float>{1, 2, 4, 8, 16, 32},
      std::vector<float>{2, 3, 5, 7, 11, 13},
      std::vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13},
      true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise_plain_real_unpacked) {
  concat_test(
      ngraph::Shape{2, 2}, ngraph::Shape{3, 2}, ngraph::Shape{3, 2}, 0,
      std::vector<float>{2, 4, 8, 16}, std::vector<float>{1, 2, 4, 8, 16, 32},
      std::vector<float>{2, 3, 5, 7, 11, 13},
      std::vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13},
      false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise_plain_real_packed) {
  concat_test(
      ngraph::Shape{2, 2}, ngraph::Shape{3, 2}, ngraph::Shape{3, 2}, 0,
      std::vector<float>{2, 4, 8, 16}, std::vector<float>{1, 2, 4, 8, 16, 32},
      std::vector<float>{2, 3, 5, 7, 11, 13},
      std::vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13},
      false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise_plain_complex_unpacked) {
  concat_test(
      ngraph::Shape{2, 2}, ngraph::Shape{3, 2}, ngraph::Shape{3, 2}, 0,
      std::vector<float>{2, 4, 8, 16}, std::vector<float>{1, 2, 4, 8, 16, 32},
      std::vector<float>{2, 3, 5, 7, 11, 13},
      std::vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13},
      false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise_plain_complex_packed) {
  concat_test(
      ngraph::Shape{2, 2}, ngraph::Shape{3, 2}, ngraph::Shape{3, 2}, 0,
      std::vector<float>{2, 4, 8, 16}, std::vector<float>{1, 2, 4, 8, 16, 32},
      std::vector<float>{2, 3, 5, 7, 11, 13},
      std::vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13},
      false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise_cipher_real_unpacked) {
  concat_test(
      ngraph::Shape{2, 2}, ngraph::Shape{3, 2}, ngraph::Shape{3, 2}, 0,
      std::vector<float>{2, 4, 8, 16}, std::vector<float>{1, 2, 4, 8, 16, 32},
      std::vector<float>{2, 3, 5, 7, 11, 13},
      std::vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13},
      true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise_cipher_real_packed) {
  concat_test(
      ngraph::Shape{2, 2}, ngraph::Shape{3, 2}, ngraph::Shape{3, 2}, 0,
      std::vector<float>{2, 4, 8, 16}, std::vector<float>{1, 2, 4, 8, 16, 32},
      std::vector<float>{2, 3, 5, 7, 11, 13},
      std::vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13},
      true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise_cipher_complex_unpacked) {
  concat_test(
      ngraph::Shape{2, 2}, ngraph::Shape{3, 2}, ngraph::Shape{3, 2}, 0,
      std::vector<float>{2, 4, 8, 16}, std::vector<float>{1, 2, 4, 8, 16, 32},
      std::vector<float>{2, 3, 5, 7, 11, 13},
      std::vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13},
      true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise_cipher_complex_packed) {
  concat_test(
      ngraph::Shape{2, 2}, ngraph::Shape{3, 2}, ngraph::Shape{3, 2}, 0,
      std::vector<float>{2, 4, 8, 16}, std::vector<float>{1, 2, 4, 8, 16, 32},
      std::vector<float>{2, 3, 5, 7, 11, 13},
      std::vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13},
      true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector_plain_real_unpacked) {
  concat_test(ngraph::Shape{4}, ngraph::Shape{6}, ngraph::Shape{2}, 0,
              std::vector<float>{2, 4, 8, 16},
              std::vector<float>{1, 2, 4, 8, 16, 32},
              std::vector<float>{18, 19},
              std::vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19},
              false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector_plain_real_packed) {
  concat_test(ngraph::Shape{4}, ngraph::Shape{6}, ngraph::Shape{2}, 0,
              std::vector<float>{2, 4, 8, 16},
              std::vector<float>{1, 2, 4, 8, 16, 32},
              std::vector<float>{18, 19},
              std::vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19},
              false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector_plain_complex_unpacked) {
  concat_test(ngraph::Shape{4}, ngraph::Shape{6}, ngraph::Shape{2}, 0,
              std::vector<float>{2, 4, 8, 16},
              std::vector<float>{1, 2, 4, 8, 16, 32},
              std::vector<float>{18, 19},
              std::vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19},
              false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector_plain_complex_packed) {
  concat_test(ngraph::Shape{4}, ngraph::Shape{6}, ngraph::Shape{2}, 0,
              std::vector<float>{2, 4, 8, 16},
              std::vector<float>{1, 2, 4, 8, 16, 32},
              std::vector<float>{18, 19},
              std::vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19},
              false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector_cipher_real_unpacked) {
  concat_test(ngraph::Shape{4}, ngraph::Shape{6}, ngraph::Shape{2}, 0,
              std::vector<float>{2, 4, 8, 16},
              std::vector<float>{1, 2, 4, 8, 16, 32},
              std::vector<float>{18, 19},
              std::vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}, true,
              false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector_cipher_real_packed) {
  concat_test(ngraph::Shape{4}, ngraph::Shape{6}, ngraph::Shape{2}, 0,
              std::vector<float>{2, 4, 8, 16},
              std::vector<float>{1, 2, 4, 8, 16, 32},
              std::vector<float>{18, 19},
              std::vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}, true,
              false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector_cipher_complex_unpacked) {
  concat_test(ngraph::Shape{4}, ngraph::Shape{6}, ngraph::Shape{2}, 0,
              std::vector<float>{2, 4, 8, 16},
              std::vector<float>{1, 2, 4, 8, 16, 32},
              std::vector<float>{18, 19},
              std::vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}, true,
              true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector_cipher_complex_packed) {
  concat_test(ngraph::Shape{4}, ngraph::Shape{6}, ngraph::Shape{2}, 0,
              std::vector<float>{2, 4, 8, 16},
              std::vector<float>{1, 2, 4, 8, 16, 32},
              std::vector<float>{18, 19},
              std::vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}, true,
              true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor_plain_real_unpacked) {
  concat_test(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1},
              ngraph::Shape{1, 1, 1, 1}, 0, std::vector<float>{1},
              std::vector<float>{2}, std::vector<float>{3},
              std::vector<float>{1, 2, 3}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor_plain_real_packed) {
  concat_test(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1},
              ngraph::Shape{1, 1, 1, 1}, 0, std::vector<float>{1},
              std::vector<float>{2}, std::vector<float>{3},
              std::vector<float>{1, 2, 3}, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor_plain_complex_unpacked) {
  concat_test(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1},
              ngraph::Shape{1, 1, 1, 1}, 0, std::vector<float>{1},
              std::vector<float>{2}, std::vector<float>{3},
              std::vector<float>{1, 2, 3}, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor_plain_complex_packed) {
  concat_test(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1},
              ngraph::Shape{1, 1, 1, 1}, 0, std::vector<float>{1},
              std::vector<float>{2}, std::vector<float>{3},
              std::vector<float>{1, 2, 3}, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor_cipher_real_unpacked) {
  concat_test(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1},
              ngraph::Shape{1, 1, 1, 1}, 0, std::vector<float>{1},
              std::vector<float>{2}, std::vector<float>{3},
              std::vector<float>{1, 2, 3}, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor_cipher_real_packed) {
  concat_test(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1},
              ngraph::Shape{1, 1, 1, 1}, 0, std::vector<float>{1},
              std::vector<float>{2}, std::vector<float>{3},
              std::vector<float>{1, 2, 3}, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor_cipher_complex_unpacked) {
  concat_test(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1},
              ngraph::Shape{1, 1, 1, 1}, 0, std::vector<float>{1},
              std::vector<float>{2}, std::vector<float>{3},
              std::vector<float>{1, 2, 3}, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor_cipher_complex_packed) {
  concat_test(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1},
              ngraph::Shape{1, 1, 1, 1}, 0, std::vector<float>{1},
              std::vector<float>{2}, std::vector<float>{3},
              std::vector<float>{1, 2, 3}, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor_plain_real_unpacked) {
  concat_test(ngraph::Shape{1, 1}, ngraph::Shape{1, 1}, ngraph::Shape{1, 1}, 0,
              std::vector<float>{1}, std::vector<float>{2},
              std::vector<float>{3}, std::vector<float>{1, 2, 3}, false, false,
              false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor_plain_real_packed) {
  concat_test(ngraph::Shape{1, 1}, ngraph::Shape{1, 1}, ngraph::Shape{1, 1}, 0,
              std::vector<float>{1}, std::vector<float>{2},
              std::vector<float>{3}, std::vector<float>{1, 2, 3}, false, false,
              true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor_plain_complex_unpacked) {
  concat_test(ngraph::Shape{1, 1}, ngraph::Shape{1, 1}, ngraph::Shape{1, 1}, 0,
              std::vector<float>{1}, std::vector<float>{2},
              std::vector<float>{3}, std::vector<float>{1, 2, 3}, false, true,
              false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor_plain_complex_packed) {
  concat_test(ngraph::Shape{1, 1}, ngraph::Shape{1, 1}, ngraph::Shape{1, 1}, 0,
              std::vector<float>{1}, std::vector<float>{2},
              std::vector<float>{3}, std::vector<float>{1, 2, 3}, false, true,
              true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor_cipher_real_unpacked) {
  concat_test(ngraph::Shape{1, 1}, ngraph::Shape{1, 1}, ngraph::Shape{1, 1}, 0,
              std::vector<float>{1}, std::vector<float>{2},
              std::vector<float>{3}, std::vector<float>{1, 2, 3}, true, false,
              false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor_cipher_real_packed) {
  concat_test(ngraph::Shape{1, 1}, ngraph::Shape{1, 1}, ngraph::Shape{1, 1}, 0,
              std::vector<float>{1}, std::vector<float>{2},
              std::vector<float>{3}, std::vector<float>{1, 2, 3}, true, false,
              true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor_cipher_complex_unpacked) {
  concat_test(ngraph::Shape{1, 1}, ngraph::Shape{1, 1}, ngraph::Shape{1, 1}, 0,
              std::vector<float>{1}, std::vector<float>{2},
              std::vector<float>{3}, std::vector<float>{1, 2, 3}, true, true,
              false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor_cipher_complex_packed) {
  concat_test(ngraph::Shape{1, 1}, ngraph::Shape{1, 1}, ngraph::Shape{1, 1}, 0,
              std::vector<float>{1}, std::vector<float>{2},
              std::vector<float>{3}, std::vector<float>{1, 2, 3}, true, true,
              true);
}

NGRAPH_TEST(${BACKEND_NAME},
            concat_zero_length_1d_middle_cipher_complex_unpacked) {
  concat_test(ngraph::Shape{4}, ngraph::Shape{0}, ngraph::Shape{4}, 0,
              std::vector<float>{1, 2, 3, 4}, std::vector<float>{},
              std::vector<float>{5, 6, 7, 8},
              std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, true, true, false);
}
