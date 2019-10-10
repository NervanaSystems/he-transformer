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
#include "ngraph/axis_set.hpp"
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

auto sum_test = [](const ngraph::Shape& in_shape,
                   const ngraph::AxisSet& axis_set,
                   const std::vector<float>& input,
                   const std::vector<float>& output, const bool arg1_encrypted,
                   const bool complex_packing, const bool packed) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        ngraph::he::HESealEncryptionParameters::
            default_complex_packing_parms());
  }

  auto a = make_shared<op::Parameter>(element::f32, in_shape);
  auto t = make_shared<op::Sum>(a, axis_set);
  auto f = make_shared<Function>(t, ParameterVector{a});

  auto annotation_from_flags = [](bool is_encrypted, bool is_packed) {
    if (is_encrypted && is_packed) {
      return HEOpAnnotations::server_ciphertext_packed_annotation();
    } else if (is_encrypted && !is_packed) {
      return HEOpAnnotations::server_ciphertext_unpacked_annotation();
    } else if (!is_encrypted && is_packed) {
      return HEOpAnnotations::server_plaintext_packed_annotation();
    } else if (!is_encrypted && !is_packed) {
      return HEOpAnnotations::server_ciphertext_unpacked_annotation();
    }
    throw ngraph_error("Logic error");
  };

  a->set_op_annotations(annotation_from_flags(arg1_encrypted, packed));

  auto tensor_from_flags = [&](const ngraph::Shape& shape, bool encrypted) {
    if (encrypted && packed) {
      return he_backend->create_packed_cipher_tensor(element::f32, shape);
    } else if (encrypted && !packed) {
      return he_backend->create_cipher_tensor(element::f32, shape);
    } else if (!encrypted && packed) {
      return he_backend->create_packed_plain_tensor(element::f32, shape);
    } else if (!encrypted && !packed) {
      return he_backend->create_plain_tensor(element::f32, shape);
    }
    throw ngraph_error("Logic error");
  };

  auto t_a = tensor_from_flags(in_shape, arg1_encrypted);
  auto t_result = tensor_from_flags(t->get_shape(), arg1_encrypted);
  copy_data(t_a, input);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a});

  EXPECT_TRUE(all_close(read_vector<float>(t_result), output, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, sum_trivial) {
  sum_test(ngraph::Shape{2, 2}, ngraph::AxisSet{},
           std::vector<float>{1, 2, 3, 4}, std::vector<float>{1, 2, 3, 4},
           false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_trivial_encrypted) {
  sum_test(ngraph::Shape{2, 2}, ngraph::AxisSet{},
           std::vector<float>{1, 2, 3, 4}, std::vector<float>{1, 2, 3, 4}, true,
           false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_trivial_5d) {
  sum_test(ngraph::Shape{2, 2, 2, 2, 2}, ngraph::AxisSet{},
           std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
           std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
           false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_trivial_5d_encrypted) {
  sum_test(ngraph::Shape{2, 2, 2, 2, 2}, ngraph::AxisSet{},
           std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
           std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
           true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_to_scalar) {
  sum_test(ngraph::Shape{2, 2}, ngraph::AxisSet{0, 1},
           std::vector<float>{1, 2, 3, 4}, std::vector<float>{10}, false, false,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_to_scalar_encrypted) {
  sum_test(ngraph::Shape{2, 2}, ngraph::AxisSet{0, 1},
           std::vector<float>{1, 2, 3, 4}, std::vector<float>{10}, true, false,
           false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_columns) {
  sum_test(ngraph::Shape{3, 2}, ngraph::AxisSet{0},
           std::vector<float>{1, 2, 3, 4, 5, 6}, std::vector<float>{9, 12},
           false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_columns_encrypted) {
  sum_test(ngraph::Shape{3, 2}, ngraph::AxisSet{0},
           std::vector<float>{1, 2, 3, 4, 5, 6}, std::vector<float>{9, 12},
           true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_rows) {
  sum_test(ngraph::Shape{3, 2}, ngraph::AxisSet{1},
           std::vector<float>{1, 2, 3, 4, 5, 6}, std::vector<float>{3, 7, 11},
           false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_rows_encrypted) {
  sum_test(ngraph::Shape{3, 2}, ngraph::AxisSet{1},
           std::vector<float>{1, 2, 3, 4, 5, 6}, std::vector<float>{3, 7, 11},
           true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_rows_zero) {
  sum_test(ngraph::Shape{3, 0}, ngraph::AxisSet{1}, std::vector<float>{},
           std::vector<float>{0, 0, 0}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_rows_zero_encrypted) {
  sum_test(ngraph::Shape{3, 0}, ngraph::AxisSet{1}, std::vector<float>{},
           std::vector<float>{0, 0, 0}, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_cols_zero) {
  sum_test(ngraph::Shape{0, 2}, ngraph::AxisSet{0}, std::vector<float>{},
           std::vector<float>{0, 0}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_cols_zero_encrypted) {
  sum_test(ngraph::Shape{0, 2}, ngraph::AxisSet{0}, std::vector<float>{},
           std::vector<float>{0, 0}, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_vector_zero) {
  sum_test(ngraph::Shape{0}, ngraph::AxisSet{0}, std::vector<float>{},
           std::vector<float>{0}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_vector_zero_encrypted) {
  sum_test(ngraph::Shape{0}, ngraph::AxisSet{0}, std::vector<float>{},
           std::vector<float>{0}, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_to_scalar_zero_by_zero) {
  sum_test(ngraph::Shape{0, 0}, ngraph::AxisSet{0, 1}, std::vector<float>{},
           std::vector<float>{0}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_to_scalar_zero_by_zero_encrypted) {
  sum_test(ngraph::Shape{0, 0}, ngraph::AxisSet{0, 1}, std::vector<float>{},
           std::vector<float>{0}, true, false, false);
}