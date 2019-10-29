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

auto dot_test = [](const ngraph::Shape& shape_a, const ngraph::Shape& shape_b,
                   const vector<float>& input_a, const vector<float>& input_b,
                   const vector<float>& output, const bool arg1_encrypted,
                   const bool arg2_encrypted, const bool complex_packing,
                   const bool packed) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        ngraph::he::HESealEncryptionParameters::
            default_complex_packing_parms());
  }

  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto b = make_shared<op::Parameter>(element::f32, shape_b);
  auto t = make_shared<op::Dot>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  NGRAPH_INFO << "arg1_encrypted " << arg1_encrypted;
  NGRAPH_INFO << "arg2_encrypted " << arg2_encrypted;
  NGRAPH_INFO << "complex_packing " << complex_packing;
  NGRAPH_INFO << "packed " << packed;

  a->set_op_annotations(
      test::he::annotation_from_flags(false, arg1_encrypted, packed));
  b->set_op_annotations(
      test::he::annotation_from_flags(false, arg2_encrypted, packed));

  auto t_a =
      test::he::tensor_from_flags(*he_backend, shape_a, arg1_encrypted, packed);
  auto t_b =
      test::he::tensor_from_flags(*he_backend, shape_b, arg2_encrypted, packed);
  auto t_result = test::he::tensor_from_flags(
      *he_backend, t->get_shape(), arg1_encrypted || arg2_encrypted, packed);

  copy_data(t_a, input_a);
  copy_data(t_b, input_b);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a, t_b});
  EXPECT_TRUE(test::he::all_close(read_vector<float>(t_result), output, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, dot1d_plain_plain_real_unpacked) {
  dot_test(Shape{4}, Shape{4}, vector<float>{2, 2, 3, 4},
           vector<float>{5, 6, 7, 8}, vector<float>{75}, false, false, false,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_plain_plain_complex_unpacked) {
  dot_test(Shape{4}, Shape{4}, vector<float>{2, 2, 3, 4},
           vector<float>{5, 6, 7, 8}, vector<float>{75}, false, false, true,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_plain_cipher_real_unpacked) {
  dot_test(Shape{4}, Shape{4}, vector<float>{2, 2, 3, 4},
           vector<float>{5, 6, 7, 8}, vector<float>{75}, false, true, false,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_plain_cipher_complex_unpacked) {
  dot_test(Shape{4}, Shape{4}, vector<float>{2, 2, 3, 4},
           vector<float>{5, 6, 7, 8}, vector<float>{75}, false, true, true,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_cipher_plain_real_unpacked) {
  dot_test(Shape{4}, Shape{4}, vector<float>{2, 2, 3, 4},
           vector<float>{5, 6, 7, 8}, vector<float>{75}, true, false, false,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_cipher_plain_complex_unpacked) {
  dot_test(Shape{4}, Shape{4}, vector<float>{2, 2, 3, 4},
           vector<float>{5, 6, 7, 8}, vector<float>{75}, true, false, true,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_cipher_cipher_real_unpacked) {
  dot_test(Shape{4}, Shape{4}, vector<float>{2, 2, 3, 4},
           vector<float>{5, 6, 7, 8}, vector<float>{75}, true, true, false,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_cipher_cipher_complex_unpacked) {
  dot_test(Shape{4}, Shape{4}, vector<float>{2, 2, 3, 4},
           vector<float>{5, 6, 7, 8}, vector<float>{75}, true, true, true,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized_plain_plain_real_unpacked) {
  dot_test(Shape{4}, Shape{4}, vector<float>{1, 2, 3, 4},
           vector<float>{-1, 0, 1, 2}, vector<float>{10}, false, false, false,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized_plain_plain_complex_unpacked) {
  dot_test(Shape{4}, Shape{4}, vector<float>{1, 2, 3, 4},
           vector<float>{-1, 0, 1, 2}, vector<float>{10}, false, false, true,
           false);
}
NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized_plain_cipher_real_unpacked) {
  dot_test(Shape{4}, Shape{4}, vector<float>{1, 2, 3, 4},
           vector<float>{-1, 0, 1, 2}, vector<float>{10}, false, true, false,
           false);
}
NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized_plain_cipher_complex_unpacked) {
  dot_test(Shape{4}, Shape{4}, vector<float>{1, 2, 3, 4},
           vector<float>{-1, 0, 1, 2}, vector<float>{10}, false, true, true,
           false);
}
NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized_cipher_plain_real_unpacked) {
  dot_test(Shape{4}, Shape{4}, vector<float>{1, 2, 3, 4},
           vector<float>{-1, 0, 1, 2}, vector<float>{10}, true, false, false,
           false);
}
NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized_cipher_plain_complex_unpacked) {
  dot_test(Shape{4}, Shape{4}, vector<float>{1, 2, 3, 4},
           vector<float>{-1, 0, 1, 2}, vector<float>{10}, true, false, true,
           false);
}
NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized_cipher_cipher_real_unpacked) {
  dot_test(Shape{4}, Shape{4}, vector<float>{1, 2, 3, 4},
           vector<float>{-1, 0, 1, 2}, vector<float>{10}, true, true, false,
           false);
}
NGRAPH_TEST(${BACKEND_NAME}, dot1d_optimized_cipher_cipher_complex_unpacked) {
  dot_test(Shape{4}, Shape{4}, vector<float>{1, 2, 3, 4},
           vector<float>{-1, 0, 1, 2}, vector<float>{10}, true, true, true,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_matrix_vector_plain_plain_real_unpacked) {
  dot_test(Shape{4, 4}, Shape{4},
           vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
           vector<float>{17, 18, 19, 20}, vector<float>{190, 486, 782, 1078},
           false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_matrix_vector_plain_plain_complex_unpacked) {
  dot_test(Shape{4, 4}, Shape{4},
           vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
           vector<float>{17, 18, 19, 20}, vector<float>{190, 486, 782, 1078},
           false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_matrix_vector_plain_cipher_real_unpacked) {
  dot_test(Shape{4, 4}, Shape{4},
           vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
           vector<float>{17, 18, 19, 20}, vector<float>{190, 486, 782, 1078},
           false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            dot1d_matrix_vector_plain_cipher_complex_unpacked) {
  dot_test(Shape{4, 4}, Shape{4},
           vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
           vector<float>{17, 18, 19, 20}, vector<float>{190, 486, 782, 1078},
           false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_matrix_vector_cipher_plain_real_unpacked) {
  dot_test(Shape{4, 4}, Shape{4},
           vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
           vector<float>{17, 18, 19, 20}, vector<float>{190, 486, 782, 1078},
           false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            dot1d_matrix_vector_cipher_plain_complex_unpacked) {
  dot_test(Shape{4, 4}, Shape{4},
           vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
           vector<float>{17, 18, 19, 20}, vector<float>{190, 486, 782, 1078},
           false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d_matrix_vector_cipher_cipher_real_unpacked) {
  dot_test(Shape{4, 4}, Shape{4},
           vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
           vector<float>{17, 18, 19, 20}, vector<float>{190, 486, 782, 1078},
           false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            dot1d_matrix_vector_cipher_cipher_complex_unpacked) {
  dot_test(Shape{4, 4}, Shape{4},
           vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
           vector<float>{17, 18, 19, 20}, vector<float>{190, 486, 782, 1078},
           false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_plain_plain_real_unpacked) {
  dot_test(Shape{}, Shape{}, vector<float>{8}, vector<float>{6},
           vector<float>{48}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_plain_plain_complex_unpacked) {
  dot_test(Shape{}, Shape{}, vector<float>{8}, vector<float>{6},
           vector<float>{48}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_plain_cipher_real_unpacked) {
  dot_test(Shape{}, Shape{}, vector<float>{8}, vector<float>{6},
           vector<float>{48}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_plain_cipher_complex_unpacked) {
  dot_test(Shape{}, Shape{}, vector<float>{8}, vector<float>{6},
           vector<float>{48}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_cipher_plain_real_unpacked) {
  dot_test(Shape{}, Shape{}, vector<float>{8}, vector<float>{6},
           vector<float>{48}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_cipher_plain_complex_unpacked) {
  dot_test(Shape{}, Shape{}, vector<float>{8}, vector<float>{6},
           vector<float>{48}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_cipher_cipher_real_unpacked) {
  dot_test(Shape{}, Shape{}, vector<float>{8}, vector<float>{6},
           vector<float>{48}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_cipher_cipher_complex_unpacked) {
  dot_test(Shape{}, Shape{}, vector<float>{8}, vector<float>{6},
           vector<float>{48}, false, false, false, false);
}
