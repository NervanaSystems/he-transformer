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

auto slice_test = [](const ngraph::Shape& shape, const ngraph::Coordinate& lower_bounds,
                     const ngraph::Coordinate& upper_bounds, const ngraph::Strides& strides,
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

  auto a = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto t = std::make_shared<ngraph::op::Slice>(a, lower_bounds, upper_bounds, strides);
  auto f = std::make_shared<ngraph::Function>(t, ngraph::ParameterVector{a});

  a->set_op_annotations(
      ngraph::test::he::annotation_from_flags(false, arg1_encrypted, packed));

  auto t_a =
      ngraph::test::he::tensor_from_flags(*he_backend, shape, arg1_encrypted, packed);
  auto t_result = ngraph::test::he::tensor_from_flags(*he_backend, t->get_shape(),
                                              arg1_encrypted, packed);

  copy_data(t_a, input);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a});
  EXPECT_TRUE(ngraph::test::he::all_close(read_vector<float>(t_result), output, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, slice_scalar) {
  for (bool arg1_encrypted : std::vector<bool>{false, true}) {
    for (bool complex_packing : std::vector<bool>{false, true}) {
      for (bool packing : std::vector<bool>{false}) {
        slice_test(ngraph::Shape{}, ngraph::Coordinate{}, ngraph::Coordinate{}, ngraph::Strides{},
                   std::vector<float>{312}, std::vector<float>{312}, arg1_encrypted,
                   complex_packing, packing);
      }
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix) {
  for (bool arg1_encrypted : std::vector<bool>{false, true}) {
    for (bool complex_packing : std::vector<bool>{false, true}) {
      for (bool packing : std::vector<bool>{false}) {
        slice_test(ngraph::Shape{4, 4}, ngraph::Coordinate{0, 1}, ngraph::Coordinate{3, 3}, ngraph::Strides{},
                   std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                 15, 16},
                   std::vector<float>{2, 3, 6, 7, 10, 11}, arg1_encrypted,
                   complex_packing, packing);
      }
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_vector) {
  for (bool arg1_encrypted : std::vector<bool>{false, true}) {
    for (bool complex_packing : std::vector<bool>{false, true}) {
      for (bool packing : std::vector<bool>{false}) {
        slice_test(
            ngraph::Shape{16}, ngraph::Coordinate{2}, ngraph::Coordinate{14}, ngraph::Strides{},
            std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
            std::vector<float>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
            arg1_encrypted, complex_packing, packing);
      }
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_strided) {
  for (bool arg1_encrypted : std::vector<bool>{false, true}) {
    for (bool complex_packing : std::vector<bool>{false, true}) {
      for (bool packing : std::vector<bool>{false}) {
        slice_test(
            ngraph::Shape{4, 4}, ngraph::Coordinate{1, 0}, ngraph::Coordinate{4, 4}, ngraph::Strides{2, 3},
            std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
            std::vector<float>{4, 7, 12, 15}, arg1_encrypted, complex_packing,
            packing);
      }
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d) {
  for (bool arg1_encrypted : std::vector<bool>{false, true}) {
    for (bool complex_packing : std::vector<bool>{false, true}) {
      for (bool packing : std::vector<bool>{false}) {
        slice_test(
            ngraph::Shape{4, 4, 4}, ngraph::Coordinate{1, 1, 1}, ngraph::Coordinate{3, 3, 3}, ngraph::Strides{},
            std::vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                          13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                          26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                          39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                          52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63},
            std::vector<float>{21, 22, 25, 26, 37, 38, 41, 42}, arg1_encrypted,
            complex_packing, packing);
      }
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d_strided) {
  for (bool arg1_encrypted : std::vector<bool>{false, true}) {
    for (bool complex_packing : std::vector<bool>{false, true}) {
      for (bool packing : std::vector<bool>{false}) {
        slice_test(
            ngraph::Shape{4, 4, 4}, ngraph::Coordinate{0, 0, 0}, ngraph::Coordinate{4, 4, 4},
            ngraph::Strides{2, 2, 2},
            std::vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                          14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                          27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                          40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                          53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64},
            std::vector<float>{1, 3, 9, 11, 33, 35, 41, 43}, arg1_encrypted,
            complex_packing, packing);
      }
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d_strided_different_strides) {
  for (bool arg1_encrypted : std::vector<bool>{false, true}) {
    for (bool complex_packing : std::vector<bool>{false, true}) {
      for (bool packing : std::vector<bool>{false}) {
        slice_test(
            ngraph::Shape{4, 4, 4}, ngraph::Coordinate{0, 0, 0}, ngraph::Coordinate{4, 4, 4},
            ngraph::Strides{2, 2, 3},
            std::vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                          14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                          27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                          40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                          53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64},
            std::vector<float>{1, 4, 9, 12, 33, 36, 41, 44}, arg1_encrypted,
            complex_packing, packing);
      }
    }
  }
}
