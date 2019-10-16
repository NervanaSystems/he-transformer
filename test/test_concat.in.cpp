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

auto concat_test = [](const Shape& shape_a, const Shape& shape_b,
                      const Shape& shape_c, size_t concat_axis,
                      const vector<float>& input_a,
                      const vector<float>& input_b,
                      const vector<float>& input_c, const vector<float>& output,
                      const bool arg1_encrypted, const bool complex_packing,
                      const bool packed) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<he::HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        he::HESealEncryptionParameters::default_complex_packing_parms());
  }

  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto b = make_shared<op::Parameter>(element::f32, shape_b);
  auto c = make_shared<op::Parameter>(element::f32, shape_c);
  auto t = make_shared<op::Concat>(NodeVector{a, b, c}, concat_axis);
  auto f = make_shared<Function>(t, ParameterVector{a, b, c});

  a->set_op_annotations(
      test::he::annotation_from_flags(false, arg1_encrypted, packed));
  b->set_op_annotations(
      test::he::annotation_from_flags(false, arg1_encrypted, packed));
  c->set_op_annotations(
      test::he::annotation_from_flags(false, arg1_encrypted, packed));

  auto t_a =
      test::he::tensor_from_flags(*he_backend, shape_a, arg1_encrypted, packed);
  auto t_b =
      test::he::tensor_from_flags(*he_backend, shape_b, arg1_encrypted, packed);
  auto t_c =
      test::he::tensor_from_flags(*he_backend, shape_c, arg1_encrypted, packed);
  auto t_result = test::he::tensor_from_flags(*he_backend, t->get_shape(),
                                              arg1_encrypted, packed);

  copy_data(t_a, input_a);
  copy_data(t_b, input_b);
  copy_data(t_c, input_c);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a, t_b, t_c});
  EXPECT_TRUE(test::he::all_close(read_vector<float>(t_result), output, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise_plain_real_unpacked) {
  concat_test(
      Shape{2, 2}, Shape{2, 3}, Shape{2, 3}, 1, vector<float>{2, 4, 8, 16},
      vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{2, 3, 5, 7, 11, 13},
      vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}, false,
      false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise_plain_real_packed) {
  concat_test(
      Shape{2, 2}, Shape{2, 3}, Shape{2, 3}, 1, vector<float>{2, 4, 8, 16},
      vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{2, 3, 5, 7, 11, 13},
      vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}, false,
      false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise_plain_complex_unpacked) {
  concat_test(
      Shape{2, 2}, Shape{2, 3}, Shape{2, 3}, 1, vector<float>{2, 4, 8, 16},
      vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{2, 3, 5, 7, 11, 13},
      vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}, false,
      true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise_plain_complex_packed) {
  concat_test(
      Shape{2, 2}, Shape{2, 3}, Shape{2, 3}, 1, vector<float>{2, 4, 8, 16},
      vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{2, 3, 5, 7, 11, 13},
      vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}, false,
      true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise_cipher_real_unpacked) {
  concat_test(
      Shape{2, 2}, Shape{2, 3}, Shape{2, 3}, 1, vector<float>{2, 4, 8, 16},
      vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{2, 3, 5, 7, 11, 13},
      vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}, true,
      false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise_cipher_real_packed) {
  concat_test(
      Shape{2, 2}, Shape{2, 3}, Shape{2, 3}, 1, vector<float>{2, 4, 8, 16},
      vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{2, 3, 5, 7, 11, 13},
      vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}, false,
      false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise_cipher_complex_unpacked) {
  concat_test(
      Shape{2, 2}, Shape{2, 3}, Shape{2, 3}, 1, vector<float>{2, 4, 8, 16},
      vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{2, 3, 5, 7, 11, 13},
      vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}, true,
      true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise_cipher_complex_packed) {
  concat_test(
      Shape{2, 2}, Shape{2, 3}, Shape{2, 3}, 1, vector<float>{2, 4, 8, 16},
      vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{2, 3, 5, 7, 11, 13},
      vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}, true,
      true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise_plain_real_unpacked) {
  concat_test(
      Shape{2, 2}, Shape{3, 2}, Shape{3, 2}, 0, vector<float>{2, 4, 8, 16},
      vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{2, 3, 5, 7, 11, 13},
      vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}, false,
      false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise_plain_real_packed) {
  concat_test(
      Shape{2, 2}, Shape{3, 2}, Shape{3, 2}, 0, vector<float>{2, 4, 8, 16},
      vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{2, 3, 5, 7, 11, 13},
      vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}, false,
      false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise_plain_complex_unpacked) {
  concat_test(
      Shape{2, 2}, Shape{3, 2}, Shape{3, 2}, 0, vector<float>{2, 4, 8, 16},
      vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{2, 3, 5, 7, 11, 13},
      vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}, false,
      true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise_plain_complex_packed) {
  concat_test(
      Shape{2, 2}, Shape{3, 2}, Shape{3, 2}, 0, vector<float>{2, 4, 8, 16},
      vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{2, 3, 5, 7, 11, 13},
      vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}, false,
      true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise_cipher_real_unpacked) {
  concat_test(
      Shape{2, 2}, Shape{3, 2}, Shape{3, 2}, 0, vector<float>{2, 4, 8, 16},
      vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{2, 3, 5, 7, 11, 13},
      vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}, true,
      false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise_cipher_real_packed) {
  concat_test(
      Shape{2, 2}, Shape{3, 2}, Shape{3, 2}, 0, vector<float>{2, 4, 8, 16},
      vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{2, 3, 5, 7, 11, 13},
      vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}, true,
      false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise_cipher_complex_unpacked) {
  concat_test(
      Shape{2, 2}, Shape{3, 2}, Shape{3, 2}, 0, vector<float>{2, 4, 8, 16},
      vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{2, 3, 5, 7, 11, 13},
      vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}, true,
      true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowise_cipher_complex_packed) {
  concat_test(
      Shape{2, 2}, Shape{3, 2}, Shape{3, 2}, 0, vector<float>{2, 4, 8, 16},
      vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{2, 3, 5, 7, 11, 13},
      vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}, true,
      true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector_plain_real_unpacked) {
  concat_test(Shape{4}, Shape{6}, Shape{2}, 0, vector<float>{2, 4, 8, 16},
              vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{18, 19},
              vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}, false,
              false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector_plain_real_packed) {
  concat_test(Shape{4}, Shape{6}, Shape{2}, 0, vector<float>{2, 4, 8, 16},
              vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{18, 19},
              vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}, false,
              false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector_plain_complex_unpacked) {
  concat_test(Shape{4}, Shape{6}, Shape{2}, 0, vector<float>{2, 4, 8, 16},
              vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{18, 19},
              vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}, false,
              true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector_plain_complex_packed) {
  concat_test(Shape{4}, Shape{6}, Shape{2}, 0, vector<float>{2, 4, 8, 16},
              vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{18, 19},
              vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}, false,
              true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector_cipher_real_unpacked) {
  concat_test(Shape{4}, Shape{6}, Shape{2}, 0, vector<float>{2, 4, 8, 16},
              vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{18, 19},
              vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}, true,
              false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector_cipher_real_packed) {
  concat_test(Shape{4}, Shape{6}, Shape{2}, 0, vector<float>{2, 4, 8, 16},
              vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{18, 19},
              vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}, true,
              false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector_cipher_complex_unpacked) {
  concat_test(Shape{4}, Shape{6}, Shape{2}, 0, vector<float>{2, 4, 8, 16},
              vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{18, 19},
              vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}, true,
              true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector_cipher_complex_packed) {
  concat_test(Shape{4}, Shape{6}, Shape{2}, 0, vector<float>{2, 4, 8, 16},
              vector<float>{1, 2, 4, 8, 16, 32}, vector<float>{18, 19},
              vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}, true,
              true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor_plain_real_unpacked) {
  concat_test(Shape{1, 1, 1, 1}, Shape{1, 1, 1, 1}, Shape{1, 1, 1, 1}, 0,
              vector<float>{1}, vector<float>{2}, vector<float>{3},
              vector<float>{1, 2, 3}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor_plain_real_packed) {
  concat_test(Shape{1, 1, 1, 1}, Shape{1, 1, 1, 1}, Shape{1, 1, 1, 1}, 0,
              vector<float>{1}, vector<float>{2}, vector<float>{3},
              vector<float>{1, 2, 3}, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor_plain_complex_unpacked) {
  concat_test(Shape{1, 1, 1, 1}, Shape{1, 1, 1, 1}, Shape{1, 1, 1, 1}, 0,
              vector<float>{1}, vector<float>{2}, vector<float>{3},
              vector<float>{1, 2, 3}, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor_plain_complex_packed) {
  concat_test(Shape{1, 1, 1, 1}, Shape{1, 1, 1, 1}, Shape{1, 1, 1, 1}, 0,
              vector<float>{1}, vector<float>{2}, vector<float>{3},
              vector<float>{1, 2, 3}, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor_cipher_real_unpacked) {
  concat_test(Shape{1, 1, 1, 1}, Shape{1, 1, 1, 1}, Shape{1, 1, 1, 1}, 0,
              vector<float>{1}, vector<float>{2}, vector<float>{3},
              vector<float>{1, 2, 3}, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor_cipher_real_packed) {
  concat_test(Shape{1, 1, 1, 1}, Shape{1, 1, 1, 1}, Shape{1, 1, 1, 1}, 0,
              vector<float>{1}, vector<float>{2}, vector<float>{3},
              vector<float>{1, 2, 3}, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor_cipher_complex_unpacked) {
  concat_test(Shape{1, 1, 1, 1}, Shape{1, 1, 1, 1}, Shape{1, 1, 1, 1}, 0,
              vector<float>{1}, vector<float>{2}, vector<float>{3},
              vector<float>{1, 2, 3}, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor_cipher_complex_packed) {
  concat_test(Shape{1, 1, 1, 1}, Shape{1, 1, 1, 1}, Shape{1, 1, 1, 1}, 0,
              vector<float>{1}, vector<float>{2}, vector<float>{3},
              vector<float>{1, 2, 3}, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor_plain_real_unpacked) {
  concat_test(Shape{1, 1}, Shape{1, 1}, Shape{1, 1}, 0, vector<float>{1},
              vector<float>{2}, vector<float>{3}, vector<float>{1, 2, 3}, false,
              false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor_plain_real_packed) {
  concat_test(Shape{1, 1}, Shape{1, 1}, Shape{1, 1}, 0, vector<float>{1},
              vector<float>{2}, vector<float>{3}, vector<float>{1, 2, 3}, false,
              false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor_plain_complex_unpacked) {
  concat_test(Shape{1, 1}, Shape{1, 1}, Shape{1, 1}, 0, vector<float>{1},
              vector<float>{2}, vector<float>{3}, vector<float>{1, 2, 3}, false,
              true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor_plain_complex_packed) {
  concat_test(Shape{1, 1}, Shape{1, 1}, Shape{1, 1}, 0, vector<float>{1},
              vector<float>{2}, vector<float>{3}, vector<float>{1, 2, 3}, false,
              true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor_cipher_real_unpacked) {
  concat_test(Shape{1, 1}, Shape{1, 1}, Shape{1, 1}, 0, vector<float>{1},
              vector<float>{2}, vector<float>{3}, vector<float>{1, 2, 3}, true,
              false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor_cipher_real_packed) {
  concat_test(Shape{1, 1}, Shape{1, 1}, Shape{1, 1}, 0, vector<float>{1},
              vector<float>{2}, vector<float>{3}, vector<float>{1, 2, 3}, true,
              false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor_cipher_complex_unpacked) {
  concat_test(Shape{1, 1}, Shape{1, 1}, Shape{1, 1}, 0, vector<float>{1},
              vector<float>{2}, vector<float>{3}, vector<float>{1, 2, 3}, true,
              true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor_cipher_complex_packed) {
  concat_test(Shape{1, 1}, Shape{1, 1}, Shape{1, 1}, 0, vector<float>{1},
              vector<float>{2}, vector<float>{3}, vector<float>{1, 2, 3}, true,
              true, true);
}
