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

static std::string s_manifest = "${MANIFEST}";

auto sum_test = [](const ngraph::Shape& in_shape,
                   const ngraph::AxisSet& axis_set,
                   const std::vector<float>& input,
                   const std::vector<float>& output, const bool arg1_encrypted,
                   const bool complex_packing, const bool packed) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        ngraph::he::HESealEncryptionParameters::
            default_complex_packing_parms());
  }

  auto a =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, in_shape);
  auto t = std::make_shared<ngraph::op::Sum>(a, axis_set);
  auto f = std::make_shared<ngraph::Function>(t, ngraph::ParameterVector{a});

  a->set_op_annotations(
      ngraph::test::he::annotation_from_flags(false, arg1_encrypted, packed));

  auto t_a = ngraph::test::he::tensor_from_flags(*he_backend, in_shape,
                                                 arg1_encrypted, packed);
  auto t_result = ngraph::test::he::tensor_from_flags(
      *he_backend, t->get_shape(), arg1_encrypted, packed);
  copy_data(t_a, input);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a});

  EXPECT_TRUE(
      ngraph::test::he::all_close(read_vector<float>(t_result), output, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, sum_trivial) {
  for (bool arg1_encrypted : std::vector<bool>{false, true}) {
    for (bool complex_packing : std::vector<bool>{false, true}) {
      for (bool packing : std::vector<bool>{false}) {
        sum_test(ngraph::Shape{2, 2}, ngraph::AxisSet{},
                 std::vector<float>{1, 2, 3, 4}, std::vector<float>{1, 2, 3, 4},
                 arg1_encrypted, complex_packing, packing);
      }
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_trivial_5d) {
  for (bool arg1_encrypted : std::vector<bool>{false, true}) {
    for (bool complex_packing : std::vector<bool>{false, true}) {
      for (bool packing : std::vector<bool>{false}) {
        sum_test(
            ngraph::Shape{2, 2, 2, 2, 2}, ngraph::AxisSet{},
            std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            arg1_encrypted, complex_packing, packing);
      }
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_to_scalar) {
  for (bool arg1_encrypted : std::vector<bool>{false, true}) {
    for (bool complex_packing : std::vector<bool>{false, true}) {
      for (bool packing : std::vector<bool>{false}) {
        sum_test(ngraph::Shape{2, 2}, ngraph::AxisSet{0, 1},
                 std::vector<float>{1, 2, 3, 4}, std::vector<float>{10},
                 arg1_encrypted, complex_packing, packing);
      }
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_columns) {
  for (bool arg1_encrypted : std::vector<bool>{false, true}) {
    for (bool complex_packing : std::vector<bool>{false, true}) {
      for (bool packing : std::vector<bool>{false}) {
        sum_test(ngraph::Shape{3, 2}, ngraph::AxisSet{0},
                 std::vector<float>{1, 2, 3, 4, 5, 6},
                 std::vector<float>{9, 12}, arg1_encrypted, complex_packing,
                 packing);
      }
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_rows) {
  for (bool arg1_encrypted : std::vector<bool>{false, true}) {
    for (bool complex_packing : std::vector<bool>{false, true}) {
      for (bool packing : std::vector<bool>{false}) {
        sum_test(ngraph::Shape{3, 2}, ngraph::AxisSet{1},
                 std::vector<float>{1, 2, 3, 4, 5, 6},
                 std::vector<float>{3, 7, 11}, arg1_encrypted, complex_packing,
                 packing);
      }
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_rows_zero) {
  for (bool arg1_encrypted : std::vector<bool>{false, true}) {
    for (bool complex_packing : std::vector<bool>{false, true}) {
      for (bool packing : std::vector<bool>{false}) {
        sum_test(ngraph::Shape{3, 0}, ngraph::AxisSet{1}, std::vector<float>{},
                 std::vector<float>{0, 0, 0}, arg1_encrypted, complex_packing,
                 packing);
      }
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_cols_zero) {
  for (bool arg1_encrypted : std::vector<bool>{false, true}) {
    for (bool complex_packing : std::vector<bool>{false, true}) {
      for (bool packing : std::vector<bool>{false}) {
        sum_test(ngraph::Shape{0, 2}, ngraph::AxisSet{0}, std::vector<float>{},
                 std::vector<float>{0, 0}, arg1_encrypted, complex_packing,
                 packing);
      }
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_vector_zero) {
  for (bool arg1_encrypted : std::vector<bool>{false, true}) {
    for (bool complex_packing : std::vector<bool>{false, true}) {
      for (bool packing : std::vector<bool>{false}) {
        sum_test(ngraph::Shape{0}, ngraph::AxisSet{0}, std::vector<float>{},
                 std::vector<float>{0}, arg1_encrypted, complex_packing,
                 packing);
      }
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_to_scalar_zero_by_zero) {
  for (bool arg1_encrypted : std::vector<bool>{false, true}) {
    for (bool complex_packing : std::vector<bool>{false, true}) {
      for (bool packing : std::vector<bool>{false}) {
        sum_test(ngraph::Shape{0, 0}, ngraph::AxisSet{0, 1},
                 std::vector<float>{}, std::vector<float>{0}, arg1_encrypted,
                 complex_packing, packing);
      }
    }
  }
}
