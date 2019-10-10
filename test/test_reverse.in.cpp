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
using namespace he;

static string s_manifest = "${MANIFEST}";

auto reverse_test = [](const Shape& shape_a, const AxisSet& axis_set,
                       const vector<float>& input, const vector<float>& output,
                       const bool arg1_encrypted, const bool complex_packing,
                       const bool packed) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<he::HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        he::HESealEncryptionParameters::default_complex_packing_parms());
  }

  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto t = make_shared<op::Reverse>(a, axis_set);
  auto f = make_shared<Function>(t, ParameterVector{a});

  auto annotation_from_flags = [](bool is_encrypted, bool is_packed) {
    if (is_encrypted && is_packed) {
      return HEOpAnnotations::server_ciphertext_packed_annotation();
    } else if (is_encrypted && !is_packed) {
      return HEOpAnnotations::server_ciphertext_unpacked_annotation();
    } else if (!is_encrypted && is_packed) {
      return HEOpAnnotations::server_plaintext_packed_annotation();
    } else if (!is_encrypted && !is_packed) {
      return HEOpAnnotations::server_plaintext_unpacked_annotation();
    }
    throw ngraph_error("Logic error");
  };

  a->set_op_annotations(annotation_from_flags(arg1_encrypted, packed));

  auto tensor_from_flags = [&](const Shape& shape, bool encrypted) {
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

  auto t_a = tensor_from_flags(shape_a, arg1_encrypted);
  auto t_result = tensor_from_flags(t->get_shape(), arg1_encrypted);

  copy_data(t_a, input);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a});
  EXPECT_TRUE(all_close(read_vector<float>(t_result), output, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, reverse_0d_plain_plain) {
  reverse_test(Shape{}, AxisSet{}, vector<float>{6}, vector<float>{6}, false,
               false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_1d_nochange_plain_plain) {
  reverse_test(Shape{8}, AxisSet{}, vector<float>{0, 1, 2, 3, 4, 5, 6, 7},
               vector<float>{0, 1, 2, 3, 4, 5, 6, 7}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_1d_0_plain_plain) {
  reverse_test(Shape{8}, AxisSet{0}, vector<float>{0, 1, 2, 3, 4, 5, 6, 7},
               vector<float>{7, 6, 5, 4, 3, 2, 1, 0}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_nochange_plain_plain) {
  reverse_test(
      Shape{4, 3}, AxisSet{},
      test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}})
          .get_vector(),
      test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}})
          .get_vector(),
      false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_0_plain_plain) {
  reverse_test(
      Shape{4, 3}, AxisSet{0},
      test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}})
          .get_vector(),
      test::NDArray<float, 2>({{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}})
          .get_vector(),
      false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_1_plain_plain) {
  reverse_test(
      Shape{4, 3}, AxisSet{1},
      test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}})
          .get_vector(),
      test::NDArray<float, 2>({{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}})
          .get_vector(),
      false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_01_plain_plain) {
  reverse_test(
      Shape{4, 3}, AxisSet{0, 1},
      test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}})
          .get_vector(),
      test::NDArray<float, 2>({{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}})
          .get_vector(),
      false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_nochange_plain_plain) {
  reverse_test(Shape{2, 4, 3}, AxisSet{},
               test::NDArray<float, 3>(
                   {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                    {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                   .get_vector(),
               test::NDArray<float, 3>(
                   {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                    {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                   .get_vector(),
               false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_0_plain_plain) {
  reverse_test(Shape{2, 4, 3}, AxisSet{0},
               test::NDArray<float, 3>(
                   {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                    {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                   .get_vector(),
               test::NDArray<float, 3>(
                   {{{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}},
                    {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}})
                   .get_vector(),
               false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_1_plain_plain) {
  reverse_test(Shape{2, 4, 3}, AxisSet{1},
               test::NDArray<float, 3>(
                   {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                    {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                   .get_vector(),
               test::NDArray<float, 3>(
                   {{{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}},
                    {{21, 22, 23}, {18, 19, 20}, {15, 16, 17}, {12, 13, 14}}})
                   .get_vector(),
               false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_2_plain_plain) {
  reverse_test(Shape{2, 4, 3}, AxisSet{2},
               test::NDArray<float, 3>(
                   {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                    {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                   .get_vector(),
               test::NDArray<float, 3>(
                   {{{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}},
                    {{14, 13, 12}, {17, 16, 15}, {20, 19, 18}, {23, 22, 21}}})
                   .get_vector(),
               false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_01_plain_plain) {
  reverse_test(Shape{2, 4, 3}, AxisSet{0, 1},
               test::NDArray<float, 3>(
                   {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                    {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                   .get_vector(),
               test::NDArray<float, 3>(
                   {{{21, 22, 23}, {18, 19, 20}, {15, 16, 17}, {12, 13, 14}},
                    {{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}}})
                   .get_vector(),
               false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_02_plain_plain) {
  reverse_test(Shape{2, 4, 3}, AxisSet{0, 2},
               test::NDArray<float, 3>(
                   {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                    {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                   .get_vector(),
               test::NDArray<float, 3>(
                   {{{14, 13, 12}, {17, 16, 15}, {20, 19, 18}, {23, 22, 21}},
                    {{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}}})
                   .get_vector(),
               false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_12_plain_plain) {
  reverse_test(Shape{2, 4, 3}, AxisSet{1, 2},
               test::NDArray<float, 3>(
                   {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                    {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                   .get_vector(),
               test::NDArray<float, 3>(
                   {{{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}},
                    {{23, 22, 21}, {20, 19, 18}, {17, 16, 15}, {14, 13, 12}}})
                   .get_vector(),
               false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_012_plain_plain) {
  reverse_test(Shape{2, 4, 3}, AxisSet{0, 1, 2},
               test::NDArray<float, 3>(
                   {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                    {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                   .get_vector(),
               test::NDArray<float, 3>(
                   {{{23, 22, 21}, {20, 19, 18}, {17, 16, 15}, {14, 13, 12}},
                    {{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}}})
                   .get_vector(),
               false, false, false);
}
