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

namespace ngraph::runtime::he {

auto dot_test = [](const Shape& shape_a, const Shape& shape_b,
                   const std::vector<float>& input_a,
                   const std::vector<float>& input_b,
                   const std::vector<float>& output, const bool arg1_encrypted,
                   const bool arg2_encrypted, const bool complex_packing,
                   const bool packed) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        HESealEncryptionParameters::default_complex_packing_parms());
  }

  auto a = std::make_shared<op::Parameter>(element::f32, shape_a);
  auto b = std::make_shared<op::Parameter>(element::f32, shape_b);
  auto t = std::make_shared<op::Dot>(a, b);
  auto f = std::make_shared<Function>(t, ParameterVector{a, b});

  const auto& arg1_config =
      test::config_from_flags(false, arg1_encrypted, packed);
  const auto& arg2_config =
      test::config_from_flags(false, arg2_encrypted, packed);

  std::string error_str;
  he_backend->set_config(
      {{a->get_name(), arg1_config}, {b->get_name(), arg2_config}}, error_str);

  auto t_a =
      test::tensor_from_flags(*he_backend, shape_a, arg1_encrypted, packed);
  auto t_b =
      test::tensor_from_flags(*he_backend, shape_b, arg2_encrypted, packed);
  auto t_result = test::tensor_from_flags(
      *he_backend, t->get_shape(), arg1_encrypted || arg2_encrypted, packed);

  copy_data(t_a, input_a);
  copy_data(t_b, input_b);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a, t_b});
  EXPECT_TRUE(test::all_close(read_vector<float>(t_result), output, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, dot1d_plain_plain_real_unpacked) {
  dot_test(Shape{4}, Shape{4}, std::vector<float>{2, 2, 3, 4},
           std::vector<float>{5, 6, 7, 8}, std::vector<float>{75}, false, false,
           false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_plain_plain_complex_unpacked) {
  dot_test(Shape{4}, Shape{4}, std::vector<float>{2, 2, 3, 4},
           std::vector<float>{5, 6, 7, 8}, std::vector<float>{75}, false, false,
           true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_plain_cipher_real_unpacked) {
  dot_test(Shape{4}, Shape{4}, std::vector<float>{2, 2, 3, 4},
           std::vector<float>{5, 6, 7, 8}, std::vector<float>{75}, false, true,
           false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_plain_cipher_complex_unpacked) {
  dot_test(Shape{4}, Shape{4}, std::vector<float>{2, 2, 3, 4},
           std::vector<float>{5, 6, 7, 8}, std::vector<float>{75}, false, true,
           true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_cipher_plain_real_unpacked) {
  dot_test(Shape{4}, Shape{4}, std::vector<float>{2, 2, 3, 4},
           std::vector<float>{5, 6, 7, 8}, std::vector<float>{75}, true, false,
           false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_cipher_plain_complex_unpacked) {
  dot_test(Shape{4}, Shape{4}, std::vector<float>{2, 2, 3, 4},
           std::vector<float>{5, 6, 7, 8}, std::vector<float>{75}, true, false,
           true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_cipher_cipher_real_unpacked) {
  dot_test(Shape{4}, Shape{4}, std::vector<float>{2, 2, 3, 4},
           std::vector<float>{5, 6, 7, 8}, std::vector<float>{75}, true, true,
           false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_cipher_cipher_complex_unpacked) {
  dot_test(Shape{4}, Shape{4}, std::vector<float>{2, 2, 3, 4},
           std::vector<float>{5, 6, 7, 8}, std::vector<float>{75}, true, true,
           true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized_plain_plain_real_unpacked) {
  dot_test(Shape{4}, Shape{4}, std::vector<float>{1, 2, 3, 4},
           std::vector<float>{-1, 0, 1, 2}, std::vector<float>{10}, false,
           false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized_plain_plain_complex_unpacked) {
  dot_test(Shape{4}, Shape{4}, std::vector<float>{1, 2, 3, 4},
           std::vector<float>{-1, 0, 1, 2}, std::vector<float>{10}, false,
           false, true, false);
}
NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized_plain_cipher_real_unpacked) {
  dot_test(Shape{4}, Shape{4}, std::vector<float>{1, 2, 3, 4},
           std::vector<float>{-1, 0, 1, 2}, std::vector<float>{10}, false, true,
           false, false);
}
NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized_plain_cipher_complex_unpacked) {
  dot_test(Shape{4}, Shape{4}, std::vector<float>{1, 2, 3, 4},
           std::vector<float>{-1, 0, 1, 2}, std::vector<float>{10}, false, true,
           true, false);
}
NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized_cipher_plain_real_unpacked) {
  dot_test(Shape{4}, Shape{4}, std::vector<float>{1, 2, 3, 4},
           std::vector<float>{-1, 0, 1, 2}, std::vector<float>{10}, true, false,
           false, false);
}
NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized_cipher_plain_complex_unpacked) {
  dot_test(Shape{4}, Shape{4}, std::vector<float>{1, 2, 3, 4},
           std::vector<float>{-1, 0, 1, 2}, std::vector<float>{10}, true, false,
           true, false);
}
NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized_cipher_cipher_real_unpacked) {
  dot_test(Shape{4}, Shape{4}, std::vector<float>{1, 2, 3, 4},
           std::vector<float>{-1, 0, 1, 2}, std::vector<float>{10}, true, true,
           false, false);
}
NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized_cipher_cipher_complex_unpacked) {
  dot_test(Shape{4}, Shape{4}, std::vector<float>{1, 2, 3, 4},
           std::vector<float>{-1, 0, 1, 2}, std::vector<float>{10}, true, true,
           true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_matrix_vector_plain_plain_real_unpacked) {
  dot_test(
      Shape{4, 4}, Shape{4},
      std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      std::vector<float>{17, 18, 19, 20},
      std::vector<float>{190, 486, 782, 1078}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_matrix_vector_plain_plain_complex_unpacked) {
  dot_test(
      Shape{4, 4}, Shape{4},
      std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      std::vector<float>{17, 18, 19, 20},
      std::vector<float>{190, 486, 782, 1078}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_matrix_vector_plain_cipher_real_unpacked) {
  dot_test(
      Shape{4, 4}, Shape{4},
      std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      std::vector<float>{17, 18, 19, 20},
      std::vector<float>{190, 486, 782, 1078}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            dot1d_matrix_vector_plain_cipher_complex_unpacked) {
  dot_test(
      Shape{4, 4}, Shape{4},
      std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      std::vector<float>{17, 18, 19, 20},
      std::vector<float>{190, 486, 782, 1078}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_matrix_vector_cipher_plain_real_unpacked) {
  dot_test(
      Shape{4, 4}, Shape{4},
      std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      std::vector<float>{17, 18, 19, 20},
      std::vector<float>{190, 486, 782, 1078}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            dot1d_matrix_vector_cipher_plain_complex_unpacked) {
  dot_test(
      Shape{4, 4}, Shape{4},
      std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      std::vector<float>{17, 18, 19, 20},
      std::vector<float>{190, 486, 782, 1078}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_matrix_vector_cipher_cipher_real_unpacked) {
  dot_test(
      Shape{4, 4}, Shape{4},
      std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      std::vector<float>{17, 18, 19, 20},
      std::vector<float>{190, 486, 782, 1078}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            dot1d_matrix_vector_cipher_cipher_complex_unpacked) {
  dot_test(
      Shape{4, 4}, Shape{4},
      std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      std::vector<float>{17, 18, 19, 20},
      std::vector<float>{190, 486, 782, 1078}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_plain_plain_real_unpacked) {
  dot_test(Shape{}, Shape{}, std::vector<float>{8}, std::vector<float>{6},
           std::vector<float>{48}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_plain_plain_complex_unpacked) {
  dot_test(Shape{}, Shape{}, std::vector<float>{8}, std::vector<float>{6},
           std::vector<float>{48}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_plain_cipher_real_unpacked) {
  dot_test(Shape{}, Shape{}, std::vector<float>{8}, std::vector<float>{6},
           std::vector<float>{48}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_plain_cipher_complex_unpacked) {
  dot_test(Shape{}, Shape{}, std::vector<float>{8}, std::vector<float>{6},
           std::vector<float>{48}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_cipher_plain_real_unpacked) {
  dot_test(Shape{}, Shape{}, std::vector<float>{8}, std::vector<float>{6},
           std::vector<float>{48}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_cipher_plain_complex_unpacked) {
  dot_test(Shape{}, Shape{}, std::vector<float>{8}, std::vector<float>{6},
           std::vector<float>{48}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_cipher_cipher_real_unpacked) {
  dot_test(Shape{}, Shape{}, std::vector<float>{8}, std::vector<float>{6},
           std::vector<float>{48}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_cipher_cipher_complex_unpacked) {
  dot_test(Shape{}, Shape{}, std::vector<float>{8}, std::vector<float>{6},
           std::vector<float>{48}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_2x0_0_cipher_cipher_complex_unpacked) {
  dot_test(Shape{2, 0}, Shape{0}, std::vector<float>{}, std::vector<float>{},
           std::vector<float>{0, 0}, false, false, false, false);
}

}  // namespace ngraph::runtime::he
