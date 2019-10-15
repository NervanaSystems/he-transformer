//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, ware
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

auto max_test = [](const Shape& shape_a, const AxisSet& axes,
                   const vector<float>& input_a, const vector<float>& output,
                   const bool arg1_encrypted, const bool complex_packing,
                   const bool packed) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        HESealEncryptionParameters::default_complex_packing_parms());
  }

  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto t = make_shared<op::Max>(a, axes);
  auto f = make_shared<Function>(t, ParameterVector{a});

  a->set_op_annotations(
      test::he::annotation_from_flags(false, arg1_encrypted, packed));

  auto t_a =
      test::he::tensor_from_flags(*he_backend, shape_a, arg1_encrypted, packed);
  auto t_result = test::he::tensor_from_flags(*he_backend, t->get_shape(),
                                              arg1_encrypted, packed);

  copy_data(t_a, input_a);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a});
  EXPECT_TRUE(test::he::all_close(read_vector<float>(t_result), output, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, max_trivial_plain_real_unpacked) {
  max_test(Shape{2, 2}, AxisSet{}, vector<float>{1, 2, 3, 4},
           vector<float>{1, 2, 3, 4}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_trivial_plain_real_packed) {
  max_test(Shape{2, 2}, AxisSet{}, vector<float>{1, 2, 3, 4},
           vector<float>{1, 2, 3, 4}, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_trivial_plain_complex_unpacked) {
  max_test(Shape{2, 2}, AxisSet{}, vector<float>{1, 2, 3, 4},
           vector<float>{1, 2, 3, 4}, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_trivial_plain_complex_packed) {
  max_test(Shape{2, 2}, AxisSet{}, vector<float>{1, 2, 3, 4},
           vector<float>{1, 2, 3, 4}, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_trivial_cipher_real_unpacked) {
  max_test(Shape{2, 2}, AxisSet{}, vector<float>{1, 2, 3, 4},
           vector<float>{1, 2, 3, 4}, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_trivial_cipher_real_packed) {
  max_test(Shape{2, 2}, AxisSet{}, vector<float>{1, 2, 3, 4},
           vector<float>{1, 2, 3, 4}, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_trivial_cipher_complex_unpacked) {
  max_test(Shape{2, 2}, AxisSet{}, vector<float>{1, 2, 3, 4},
           vector<float>{1, 2, 3, 4}, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_trivial_cipher_complex_packed) {
  max_test(Shape{2, 2}, AxisSet{}, vector<float>{1, 2, 3, 4},
           vector<float>{1, 2, 3, 4}, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_to_scalar_plain_real_unpacked) {
  max_test(Shape{2, 2}, AxisSet{0, 1}, vector<float>{1, 2, 3, 4},
           vector<float>{4}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_to_scalar_plain_real_packed) {
  max_test(Shape{2, 2}, AxisSet{0, 1}, vector<float>{1, 2, 3, 4},
           vector<float>{4}, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_to_scalar_plain_complex_unpacked) {
  max_test(Shape{2, 2}, AxisSet{0, 1}, vector<float>{1, 2, 3, 4},
           vector<float>{4}, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_to_scalar_plain_complex_packed) {
  max_test(Shape{2, 2}, AxisSet{0, 1}, vector<float>{1, 2, 3, 4},
           vector<float>{4}, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_to_scalar_cipher_real_unpacked) {
  max_test(Shape{2, 2}, AxisSet{0, 1}, vector<float>{1, 2, 3, 4},
           vector<float>{4}, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_to_scalar_cipher_real_packed) {
  max_test(Shape{2, 2}, AxisSet{0, 1}, vector<float>{1, 2, 3, 4},
           vector<float>{4}, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_to_scalar_cipher_complex_unpacked) {
  max_test(Shape{2, 2}, AxisSet{0, 1}, vector<float>{1, 2, 3, 4},
           vector<float>{4}, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_to_scalar_cipher_complex_packed) {
  max_test(Shape{2, 2}, AxisSet{0, 1}, vector<float>{1, 2, 3, 4},
           vector<float>{4}, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_columns_plain_real_unpacked) {
  max_test(Shape{3, 2}, AxisSet{0}, vector<float>{1, 2, 3, 4, 5, 6},
           vector<float>{5, 6}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_columns_plain_real_packed) {
  max_test(Shape{3, 2}, AxisSet{0}, vector<float>{1, 2, 3, 4, 5, 6},
           vector<float>{5, 6}, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_columns_plain_complex_unpacked) {
  max_test(Shape{3, 2}, AxisSet{0}, vector<float>{1, 2, 3, 4, 5, 6},
           vector<float>{5, 6}, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_columns_plain_complex_packed) {
  max_test(Shape{3, 2}, AxisSet{0}, vector<float>{1, 2, 3, 4, 5, 6},
           vector<float>{5, 6}, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_columns_cipher_real_unpacked) {
  max_test(Shape{3, 2}, AxisSet{0}, vector<float>{1, 2, 3, 4, 5, 6},
           vector<float>{5, 6}, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_columns_cipher_real_packed) {
  max_test(Shape{3, 2}, AxisSet{0}, vector<float>{1, 2, 3, 4, 5, 6},
           vector<float>{5, 6}, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_columns_cipher_complex_unpacked) {
  max_test(Shape{3, 2}, AxisSet{0}, vector<float>{1, 2, 3, 4, 5, 6},
           vector<float>{5, 6}, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_columns_cipher_complex_packed) {
  max_test(Shape{3, 2}, AxisSet{0}, vector<float>{1, 2, 3, 4, 5, 6},
           vector<float>{5, 6}, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_plain_real_unpacked) {
  max_test(Shape{3, 2}, AxisSet{1}, vector<float>{1, 2, 3, 4, 5, 6},
           vector<float>{2, 4, 6}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_plain_real_packed) {
  max_test(Shape{3, 2}, AxisSet{1}, vector<float>{1, 2, 3, 4, 5, 6},
           vector<float>{2, 4, 6}, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_plain_complex_unpacked) {
  max_test(Shape{3, 2}, AxisSet{1}, vector<float>{1, 2, 3, 4, 5, 6},
           vector<float>{2, 4, 6}, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_plain_complex_packed) {
  max_test(Shape{3, 2}, AxisSet{1}, vector<float>{1, 2, 3, 4, 5, 6},
           vector<float>{2, 4, 6}, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_cipher_real_unpacked) {
  max_test(Shape{3, 2}, AxisSet{1}, vector<float>{1, 2, 3, 4, 5, 6},
           vector<float>{2, 4, 6}, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_cipher_real_packed) {
  max_test(Shape{3, 2}, AxisSet{1}, vector<float>{1, 2, 3, 4, 5, 6},
           vector<float>{2, 4, 6}, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_cipher_complex_unpacked) {
  max_test(Shape{3, 2}, AxisSet{1}, vector<float>{1, 2, 3, 4, 5, 6},
           vector<float>{2, 4, 6}, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_cipher_complex_packed) {
  max_test(Shape{3, 2}, AxisSet{1}, vector<float>{1, 2, 3, 4, 5, 6},
           vector<float>{2, 4, 6}, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_zero_plain_real_unpacked) {
  max_test(Shape{3, 0}, AxisSet{1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()},
           false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_zero_plain_real_packed) {
  max_test(Shape{3, 0}, AxisSet{1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()},
           false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_zero_plain_complex_unpacked) {
  max_test(Shape{3, 0}, AxisSet{1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()},
           false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_zero_plain_complex_packed) {
  max_test(Shape{3, 0}, AxisSet{1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()},
           false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_zero_cipher_real_unpacked) {
  max_test(Shape{3, 0}, AxisSet{1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()},
           true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_zero_cipher_real_packed) {
  max_test(Shape{3, 0}, AxisSet{1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()},
           true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_zero_cipher_complex_unpacked) {
  max_test(Shape{3, 0}, AxisSet{1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()},
           true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_zero_cipher_complex_packed) {
  max_test(Shape{3, 0}, AxisSet{1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()},
           true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_cols_zero_plain_real_unpacked) {
  max_test(Shape{0, 2}, AxisSet{0}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()},
           false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_cols_zero_plain_real_packed) {
  max_test(Shape{0, 2}, AxisSet{0}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()},
           false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_cols_zero_plain_complex_unpacked) {
  max_test(Shape{0, 2}, AxisSet{0}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()},
           false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_cols_zero_plain_complex_packed) {
  max_test(Shape{0, 2}, AxisSet{0}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()},
           false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_cols_zero_cipher_real_unpacked) {
  max_test(Shape{0, 2}, AxisSet{0}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()},
           true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_cols_zero_cipher_real_packed) {
  max_test(Shape{0, 2}, AxisSet{0}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()},
           true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_cols_zero_cipher_complex_unpacked) {
  max_test(Shape{0, 2}, AxisSet{0}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()},
           true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_cols_zero_cipher_complex_packed) {
  max_test(Shape{0, 2}, AxisSet{0}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()},
           true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_vector_zero_plain_real_unpacked) {
  max_test(Shape{0}, AxisSet{0}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, false, false,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_vector_zero_plain_real_packed) {
  max_test(Shape{0}, AxisSet{0}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, false, false,
           true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_vector_zero_plain_complex_unpacked) {
  max_test(Shape{0}, AxisSet{0}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, false, true,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_vector_zero_plain_complex_packed) {
  max_test(Shape{0}, AxisSet{0}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, false, true,
           true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_vector_zero_cipher_real_unpacked) {
  max_test(Shape{0}, AxisSet{0}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, true, false,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_vector_zero_cipher_real_packed) {
  max_test(Shape{0}, AxisSet{0}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, true, false,
           true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_vector_zero_cipher_complex_unpacked) {
  max_test(Shape{0}, AxisSet{0}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, true, true,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_vector_zero_cipher_complex_packed) {
  max_test(Shape{0}, AxisSet{0}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, true, true,
           true);
}

NGRAPH_TEST(${BACKEND_NAME},
            max_matrix_to_scalar_zero_by_zero_plain_real_unpacked) {
  max_test(Shape{0, 0}, AxisSet{0, 1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, false, false,
           false);
}

NGRAPH_TEST(${BACKEND_NAME},
            max_matrix_to_scalar_zero_by_zero_plain_real_packed) {
  max_test(Shape{0, 0}, AxisSet{0, 1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, false, false,
           true);
}

NGRAPH_TEST(${BACKEND_NAME},
            max_matrix_to_scalar_zero_by_zero_plain_complex_unpacked) {
  max_test(Shape{0, 0}, AxisSet{0, 1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, false, true,
           false);
}

NGRAPH_TEST(${BACKEND_NAME},
            max_matrix_to_scalar_zero_by_zero_plain_complex_packed) {
  max_test(Shape{0, 0}, AxisSet{0, 1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, false, true,
           true);
}

NGRAPH_TEST(${BACKEND_NAME},
            max_matrix_to_scalar_zero_by_zero_cipher_real_unpacked) {
  max_test(Shape{0, 0}, AxisSet{0, 1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, true, false,
           false);
}

NGRAPH_TEST(${BACKEND_NAME},
            max_matrix_to_scalar_zero_by_zero_cipher_real_packed) {
  max_test(Shape{0, 0}, AxisSet{0, 1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, true, false,
           true);
}

NGRAPH_TEST(${BACKEND_NAME},
            max_matrix_to_scalar_zero_by_zero_cipher_complex_unpacked) {
  max_test(Shape{0, 0}, AxisSet{0, 1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, true, true,
           false);
}

NGRAPH_TEST(${BACKEND_NAME},
            max_matrix_to_scalar_zero_by_zero_cipher_complex_packed) {
  max_test(Shape{0, 0}, AxisSet{0, 1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, true, true,
           true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_most_sig_plain_real_unpacked) {
  max_test(Shape{0, 0}, AxisSet{0, 1}, vector<float>{},
           vector<float>{-std::numeric_limits<float>::infinity()}, false, false,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_most_sig_plain_real_packed) {
  max_test(Shape{3, 3, 3}, AxisSet{0},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{19, 20, 21, 22, 23, 24, 25, 26, 27}, false, false,
           true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_most_sig_plain_complex_unpacked) {
  max_test(Shape{3, 3, 3}, AxisSet{0},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{19, 20, 21, 22, 23, 24, 25, 26, 27}, false, true,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_most_sig_plain_complex_packed) {
  max_test(Shape{3, 3, 3}, AxisSet{0},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{19, 20, 21, 22, 23, 24, 25, 26, 27}, false, true,
           true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_most_sig_cipher_real_unpacked) {
  max_test(Shape{3, 3, 3}, AxisSet{0},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{19, 20, 21, 22, 23, 24, 25, 26, 27}, true, false,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_most_sig_cipher_real_packed) {
  max_test(Shape{3, 3, 3}, AxisSet{0},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{19, 20, 21, 22, 23, 24, 25, 26, 27}, true, false,
           true);
}

NGRAPH_TEST(${BACKEND_NAME},
            max_3d_to_matrix_most_sig_cipher_complex_unpacked) {
  max_test(Shape{3, 3, 3}, AxisSet{0},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{19, 20, 21, 22, 23, 24, 25, 26, 27}, true, true,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_most_sig_cipher_complex_packed) {
  max_test(Shape{3, 3, 3}, AxisSet{0},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{19, 20, 21, 22, 23, 24, 25, 26, 27}, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_least_sig_plain_real_unpacked) {
  max_test(Shape{3, 3, 3}, AxisSet{2},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{3, 6, 9, 12, 15, 18, 21, 24, 27}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_least_sig_plain_real_packed) {
  max_test(Shape{3, 3, 3}, AxisSet{2},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{3, 6, 9, 12, 15, 18, 21, 24, 27}, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME},
            max_3d_to_matrix_least_sig_plain_complex_unpacked) {
  max_test(Shape{3, 3, 3}, AxisSet{2},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{3, 6, 9, 12, 15, 18, 21, 24, 27}, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_least_sig_plain_complex_packed) {
  max_test(Shape{3, 3, 3}, AxisSet{2},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{3, 6, 9, 12, 15, 18, 21, 24, 27}, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_least_sig_cipher_real_unpacked) {
  max_test(Shape{3, 3, 3}, AxisSet{2},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{3, 6, 9, 12, 15, 18, 21, 24, 27}, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_least_sig_cipher_real_packed) {
  max_test(Shape{3, 3, 3}, AxisSet{2},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{3, 6, 9, 12, 15, 18, 21, 24, 27}, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME},
            max_3d_to_matrix_least_sig_cipher_complex_unpacked) {
  max_test(Shape{3, 3, 3}, AxisSet{2},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{3, 6, 9, 12, 15, 18, 21, 24, 27}, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_least_sig_cipher_complex_packed) {
  max_test(Shape{3, 3, 3}, AxisSet{2},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{3, 6, 9, 12, 15, 18, 21, 24, 27}, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_vector_plain_real_unpacked) {
  max_test(Shape{3, 3, 3}, AxisSet{0, 1},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{25, 26, 27}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_vector_plain_real_packed) {
  max_test(Shape{3, 3, 3}, AxisSet{0, 1},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{25, 26, 27}, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_vector_plain_complex_unpacked) {
  max_test(Shape{3, 3, 3}, AxisSet{0, 1},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{25, 26, 27}, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_vector_plain_complex_packed) {
  max_test(Shape{3, 3, 3}, AxisSet{0, 1},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{25, 26, 27}, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_vector_cipher_real_unpacked) {
  max_test(Shape{3, 3, 3}, AxisSet{0, 1},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{25, 26, 27}, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_vector_cipher_real_packed) {
  max_test(Shape{3, 3, 3}, AxisSet{0, 1},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{25, 26, 27}, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_vector_cipher_complex_unpacked) {
  max_test(Shape{3, 3, 3}, AxisSet{0, 1},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{25, 26, 27}, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_vector_cipher_complex_packed) {
  max_test(Shape{3, 3, 3}, AxisSet{0, 1},
           vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
           vector<float>{25, 26, 27}, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_scalar_plain_real_unpacked) {
  max_test(Shape{3, 3, 3}, AxisSet{0, 1, 2},
           vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                         13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1},
           vector<float>{14}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_scalar_plain_real_packed) {
  max_test(Shape{3, 3, 3}, AxisSet{0, 1, 2},
           vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                         13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1},
           vector<float>{14}, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_scalar_plain_complex_unpacked) {
  max_test(Shape{3, 3, 3}, AxisSet{0, 1, 2},
           vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                         13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1},
           vector<float>{14}, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_scalar_plain_complex_packed) {
  max_test(Shape{3, 3, 3}, AxisSet{0, 1, 2},
           vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                         13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1},
           vector<float>{14}, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_scalar_cipher_real_unpacked) {
  max_test(Shape{3, 3, 3}, AxisSet{0, 1, 2},
           vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                         13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1},
           vector<float>{14}, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_scalar_cipher_real_packed) {
  max_test(Shape{3, 3, 3}, AxisSet{0, 1, 2},
           vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                         13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1},
           vector<float>{14}, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_scalar_cipher_complex_unpacked) {
  max_test(Shape{3, 3, 3}, AxisSet{0, 1, 2},
           vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                         13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1},
           vector<float>{14}, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_scalar_cipher_complex_packed) {
  max_test(Shape{3, 3, 3}, AxisSet{0, 1, 2},
           vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                         13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1},
           vector<float>{14}, true, true, true);
}
