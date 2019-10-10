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

auto conv_test = [](const ngraph::Shape& shape_a, const ngraph::Shape& shape_b,
                    const Strides& window_movement_strides,
                    const Strides& window_dilation_strides,
                    const CoordinateDiff& padding_below,
                    const CoordinateDiff& padding_above,
                    const Strides& data_dilation_strides,
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
  auto t = make_shared<op::Convolution>(a, b, window_movement_strides,
                                        window_dilation_strides, padding_below,
                                        padding_above, data_dilation_strides);

  auto f = make_shared<Function>(t, ParameterVector{a, b});

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

  NGRAPH_INFO << "shape_a " << shape_a;
  NGRAPH_INFO << "shape_b " << shape_b;

  a->set_op_annotations(annotation_from_flags(arg1_encrypted, packed));
  // Weights should not be packed
  b->set_op_annotations(annotation_from_flags(arg2_encrypted, false));

  auto tensor_from_flags = [&](const Shape& shape, bool encrypted,
                               bool is_packed) {
    if (encrypted && is_packed) {
      return he_backend->create_packed_cipher_tensor(element::f32, shape);
    } else if (encrypted && !is_packed) {
      return he_backend->create_cipher_tensor(element::f32, shape);
    } else if (!encrypted && is_packed) {
      return he_backend->create_packed_plain_tensor(element::f32, shape);
    } else if (!encrypted && !is_packed) {
      return he_backend->create_plain_tensor(element::f32, shape);
    }
    throw ngraph_error("Logic error");
  };

  auto t_a = tensor_from_flags(shape_a, arg1_encrypted, packed);
  // Weights should not be packed
  auto t_b = tensor_from_flags(shape_b, arg2_encrypted, false);
  auto t_result = tensor_from_flags(t->get_shape(),
                                    arg1_encrypted | arg2_encrypted, packed);

  copy_data(t_a, input_a);
  copy_data(t_b, input_b);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a, t_b});
  EXPECT_TRUE(all_close(read_vector<float>(t_result), output, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_1image_plain_plain_real_unpacked) {
  conv_test(Shape{1, 1, 5, 5}, Shape{1, 1, 3, 3}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{}, CoordinateDiff{}, Strides{},
            vector<float>{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
            vector<float>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
            vector<float>{9, 9, 9, 9, 9, 9, 9, 9, 9}, false, false, false,
            false);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_1image_plain_plain_real_packed) {
  conv_test(Shape{1, 1, 5, 5}, Shape{1, 1, 3, 3}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{}, CoordinateDiff{}, Strides{},
            vector<float>{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
            vector<float>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
            vector<float>{9, 9, 9, 9, 9, 9, 9, 9, 9}, false, false, false,
            true);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_1image_plain_plain_complex_unpacked) {
  conv_test(Shape{1, 1, 5, 5}, Shape{1, 1, 3, 3}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{}, CoordinateDiff{}, Strides{},
            vector<float>{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
            vector<float>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
            vector<float>{9, 9, 9, 9, 9, 9, 9, 9, 9}, false, false, true,
            false);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_1image_plain_plain_complex_packed) {
  conv_test(Shape{1, 1, 5, 5}, Shape{1, 1, 3, 3}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{}, CoordinateDiff{}, Strides{},
            vector<float>{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
            vector<float>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
            vector<float>{9, 9, 9, 9, 9, 9, 9, 9, 9}, false, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_1image_cipher_plain_real_unpacked) {
  conv_test(Shape{1, 1, 5, 5}, Shape{1, 1, 3, 3}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{}, CoordinateDiff{}, Strides{},
            vector<float>{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
            vector<float>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
            vector<float>{9, 9, 9, 9, 9, 9, 9, 9, 9}, true, false, false,
            false);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_1image_cipher_plain_real_packed) {
  conv_test(Shape{1, 1, 5, 5}, Shape{1, 1, 3, 3}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{}, CoordinateDiff{}, Strides{},
            vector<float>{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
            vector<float>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
            vector<float>{9, 9, 9, 9, 9, 9, 9, 9, 9}, true, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_1image_cipher_plain_complex_unpacked) {
  conv_test(Shape{1, 1, 5, 5}, Shape{1, 1, 3, 3}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{}, CoordinateDiff{}, Strides{},
            vector<float>{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
            vector<float>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
            vector<float>{9, 9, 9, 9, 9, 9, 9, 9, 9}, true, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_1image_cipher_plain_complex_packed) {
  conv_test(Shape{1, 1, 5, 5}, Shape{1, 1, 3, 3}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{}, CoordinateDiff{}, Strides{},
            vector<float>{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
            vector<float>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
            vector<float>{9, 9, 9, 9, 9, 9, 9, 9, 9}, true, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_1image_plain_cipher_real_unpacked) {
  conv_test(Shape{1, 1, 5, 5}, Shape{1, 1, 3, 3}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{}, CoordinateDiff{}, Strides{},
            vector<float>{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
            vector<float>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
            vector<float>{9, 9, 9, 9, 9, 9, 9, 9, 9}, true, false, false,
            false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_1image_cipher_cipher_real_unpacked) {
  conv_test(Shape{1, 1, 5, 5}, Shape{1, 1, 3, 3}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{}, CoordinateDiff{}, Strides{},
            vector<float>{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
            vector<float>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
            vector<float>{9, 9, 9, 9, 9, 9, 9, 9, 9}, true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_1image_2outputs_plain_plain_real_unpacked) {
  conv_test(Shape{1, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{}, CoordinateDiff{}, Strides{},
            vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
            vector<float>{1, 2, 3, 4, 5, 6, 7, 8},
            vector<float>{51, 61, 71, 81, 101, 111, 121, 131, 115, 141, 167,
                          193, 245, 271, 297, 323},
            false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_1image_2outputs_plain_cipher_real_unpacked) {
  conv_test(Shape{1, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{}, CoordinateDiff{}, Strides{},
            vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
            vector<float>{1, 2, 3, 4, 5, 6, 7, 8},
            vector<float>{51, 61, 71, 81, 101, 111, 121, 131, 115, 141, 167,
                          193, 245, 271, 297, 323},
            false, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_1image_2outputs_cipher_plain_real_unpacked) {
  conv_test(Shape{1, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{}, CoordinateDiff{}, Strides{},
            vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
            vector<float>{1, 2, 3, 4, 5, 6, 7, 8},
            vector<float>{51, 61, 71, 81, 101, 111, 121, 131, 115, 141, 167,
                          193, 245, 271, 297, 323},
            true, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_1image_2outputs_cipher_cipher_real_unpacked) {
  conv_test(Shape{1, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{}, CoordinateDiff{}, Strides{},
            vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
            vector<float>{1, 2, 3, 4, 5, 6, 7, 8},
            vector<float>{51, 61, 71, 81, 101, 111, 121, 131, 115, 141, 167,
                          193, 245, 271, 297, 323},
            true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_1item_plain_plain_real_unpacked) {
  conv_test(Shape{1, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f, -8.f,
                          5.f, -8.f, 1.f, 2.f, 8.f, -2.f},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
            vector<float>{32.0f, -18.0f, 56.0f, 56.0f, -42.0f, -14.0f, -16.0f,
                          46.0f, -54.0f, -9.0f, -30.0f, 48.0f, 78.0f, -33.0f,
                          -123.0f, -21.0f},
            false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_1item_plain_cipher_real_unpacked) {
  conv_test(Shape{1, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f, -8.f,
                          5.f, -8.f, 1.f, 2.f, 8.f, -2.f},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
            vector<float>{32.0f, -18.0f, 56.0f, 56.0f, -42.0f, -14.0f, -16.0f,
                          46.0f, -54.0f, -9.0f, -30.0f, 48.0f, 78.0f, -33.0f,
                          -123.0f, -21.0f},
            false, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_1item_cipher_plain_real_unpacked) {
  conv_test(Shape{1, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f, -8.f,
                          5.f, -8.f, 1.f, 2.f, 8.f, -2.f},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
            vector<float>{32.0f, -18.0f, 56.0f, 56.0f, -42.0f, -14.0f, -16.0f,
                          46.0f, -54.0f, -9.0f, -30.0f, 48.0f, 78.0f, -33.0f,
                          -123.0f, -21.0f},
            true, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_1item_cipher_cipher_real_unpacked) {
  conv_test(Shape{1, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f, -8.f,
                          5.f, -8.f, 1.f, 2.f, 8.f, -2.f},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
            vector<float>{32.0f, -18.0f, 56.0f, 56.0f, -42.0f, -14.0f, -16.0f,
                          46.0f, -54.0f, -9.0f, -30.0f, 48.0f, 78.0f, -33.0f,
                          -123.0f, -21.0f},
            true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_1item_padded_1_1x1_1_plain_plain_real_unpacked) {
  conv_test(Shape{1, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{1, 1}, CoordinateDiff{1, 1}, Strides{1, 1},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f, -8.f,
                          5.f, -8.f, 1.f, 2.f, 8.f, -2.f},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
            vector<float>{
                16.0f,  28.0f,  0.0f,   20.0f,  -10.0f, -36.0f, -34.0f, 32.0f,
                -18.0f, 56.0f,  56.0f,  -92.0f, 34.0f,  -42.0f, -14.0f, -16.0f,
                46.0f,  -32.0f, -16.0f, 66.0f,  -4.0f,  0.0f,   -68.0f, 16.0f,
                24.0f,  -6.0f,  12.0f,  6.0f,   -27.0f, 0.0f,   -99.0f, -54.0f,
                -9.0f,  -30.0f, 48.0f,  81.0f,  105.0f, 78.0f,  -33.0f, -123.0f,
                -21.0f, 45.0f,  -72.0f, -63.0f, 27.0f,  90.0f,  54.0f,  -18.0f},
            false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_1item_padded_1_1x1_1_plain_cipher_real_unpacked) {
  conv_test(Shape{1, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{1, 1}, CoordinateDiff{1, 1}, Strides{1, 1},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f, -8.f,
                          5.f, -8.f, 1.f, 2.f, 8.f, -2.f},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
            vector<float>{
                16.0f,  28.0f,  0.0f,   20.0f,  -10.0f, -36.0f, -34.0f, 32.0f,
                -18.0f, 56.0f,  56.0f,  -92.0f, 34.0f,  -42.0f, -14.0f, -16.0f,
                46.0f,  -32.0f, -16.0f, 66.0f,  -4.0f,  0.0f,   -68.0f, 16.0f,
                24.0f,  -6.0f,  12.0f,  6.0f,   -27.0f, 0.0f,   -99.0f, -54.0f,
                -9.0f,  -30.0f, 48.0f,  81.0f,  105.0f, 78.0f,  -33.0f, -123.0f,
                -21.0f, 45.0f,  -72.0f, -63.0f, 27.0f,  90.0f,  54.0f,  -18.0f},
            false, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_1item_padded_1_1x1_1_cipher_plain_real_unpacked) {
  conv_test(Shape{1, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{1, 1}, CoordinateDiff{1, 1}, Strides{1, 1},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f, -8.f,
                          5.f, -8.f, 1.f, 2.f, 8.f, -2.f},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
            vector<float>{
                16.0f,  28.0f,  0.0f,   20.0f,  -10.0f, -36.0f, -34.0f, 32.0f,
                -18.0f, 56.0f,  56.0f,  -92.0f, 34.0f,  -42.0f, -14.0f, -16.0f,
                46.0f,  -32.0f, -16.0f, 66.0f,  -4.0f,  0.0f,   -68.0f, 16.0f,
                24.0f,  -6.0f,  12.0f,  6.0f,   -27.0f, 0.0f,   -99.0f, -54.0f,
                -9.0f,  -30.0f, 48.0f,  81.0f,  105.0f, 78.0f,  -33.0f, -123.0f,
                -21.0f, 45.0f,  -72.0f, -63.0f, 27.0f,  90.0f,  54.0f,  -18.0f},
            true, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_1item_padded_1_1x1_1_cipher_cipher_real_unpacked) {
  conv_test(Shape{1, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{1, 1}, CoordinateDiff{1, 1}, Strides{1, 1},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f, -8.f,
                          5.f, -8.f, 1.f, 2.f, 8.f, -2.f},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
            vector<float>{
                16.0f,  28.0f,  0.0f,   20.0f,  -10.0f, -36.0f, -34.0f, 32.0f,
                -18.0f, 56.0f,  56.0f,  -92.0f, 34.0f,  -42.0f, -14.0f, -16.0f,
                46.0f,  -32.0f, -16.0f, 66.0f,  -4.0f,  0.0f,   -68.0f, 16.0f,
                24.0f,  -6.0f,  12.0f,  6.0f,   -27.0f, 0.0f,   -99.0f, -54.0f,
                -9.0f,  -30.0f, 48.0f,  81.0f,  105.0f, 78.0f,  -33.0f, -123.0f,
                -21.0f, 45.0f,  -72.0f, -63.0f, 27.0f,  90.0f,  54.0f,  -18.0f},
            true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_1item_padded_2_3x4_5_plain_plain_real_unpacked) {
  conv_test(Shape{1, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{2, 3}, CoordinateDiff{4, 5}, Strides{1, 1},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f, -8.f,
                          5.f, -8.f, 1.f, 2.f, 8.f, -2.f},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
            vector<float>{
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   16.0f,  28.0f,
                0.0f,   20.0f,   -10.0f, -36.0f, 0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    -34.0f, 32.0f,  -18.0f, 56.0f,  56.0f,  -92.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   34.0f,  -42.0f,
                -14.0f, -16.0f,  46.0f,  -32.0f, 0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    -16.0f, 66.0f,  -4.0f,  0.0f,   -68.0f, 16.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   24.0f,  -6.0f,
                12.0f,  6.0f,    -27.0f, 0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    -99.0f, -54.0f, -9.0f,  -30.0f, 48.0f,  81.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   105.0f, 78.0f,
                -33.0f, -123.0f, -21.0f, 45.0f,  0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    -72.0f, -63.0f, 27.0f,  90.0f,  54.0f,  -18.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f},
            false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_1item_padded_2_3x4_5_plain_cipher_real_unpacked) {
  conv_test(Shape{1, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{2, 3}, CoordinateDiff{4, 5}, Strides{1, 1},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f, -8.f,
                          5.f, -8.f, 1.f, 2.f, 8.f, -2.f},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
            vector<float>{
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   16.0f,  28.0f,
                0.0f,   20.0f,   -10.0f, -36.0f, 0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    -34.0f, 32.0f,  -18.0f, 56.0f,  56.0f,  -92.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   34.0f,  -42.0f,
                -14.0f, -16.0f,  46.0f,  -32.0f, 0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    -16.0f, 66.0f,  -4.0f,  0.0f,   -68.0f, 16.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   24.0f,  -6.0f,
                12.0f,  6.0f,    -27.0f, 0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    -99.0f, -54.0f, -9.0f,  -30.0f, 48.0f,  81.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   105.0f, 78.0f,
                -33.0f, -123.0f, -21.0f, 45.0f,  0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    -72.0f, -63.0f, 27.0f,  90.0f,  54.0f,  -18.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f},
            false, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_1item_padded_2_3x4_5_cipher_plain_real_unpacked) {
  conv_test(Shape{1, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{2, 3}, CoordinateDiff{4, 5}, Strides{1, 1},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f, -8.f,
                          5.f, -8.f, 1.f, 2.f, 8.f, -2.f},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
            vector<float>{
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   16.0f,  28.0f,
                0.0f,   20.0f,   -10.0f, -36.0f, 0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    -34.0f, 32.0f,  -18.0f, 56.0f,  56.0f,  -92.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   34.0f,  -42.0f,
                -14.0f, -16.0f,  46.0f,  -32.0f, 0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    -16.0f, 66.0f,  -4.0f,  0.0f,   -68.0f, 16.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   24.0f,  -6.0f,
                12.0f,  6.0f,    -27.0f, 0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    -99.0f, -54.0f, -9.0f,  -30.0f, 48.0f,  81.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   105.0f, 78.0f,
                -33.0f, -123.0f, -21.0f, 45.0f,  0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    -72.0f, -63.0f, 27.0f,  90.0f,  54.0f,  -18.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f},
            true, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_1item_padded_2_3x4_5_cipher_cipher_real_unpacked) {
  conv_test(Shape{1, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
            CoordinateDiff{2, 3}, CoordinateDiff{4, 5}, Strides{1, 1},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f, -8.f,
                          5.f, -8.f, 1.f, 2.f, 8.f, -2.f},
            vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
            vector<float>{
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   16.0f,  28.0f,
                0.0f,   20.0f,   -10.0f, -36.0f, 0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    -34.0f, 32.0f,  -18.0f, 56.0f,  56.0f,  -92.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   34.0f,  -42.0f,
                -14.0f, -16.0f,  46.0f,  -32.0f, 0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    -16.0f, 66.0f,  -4.0f,  0.0f,   -68.0f, 16.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   24.0f,  -6.0f,
                12.0f,  6.0f,    -27.0f, 0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    -99.0f, -54.0f, -9.0f,  -30.0f, 48.0f,  81.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   105.0f, 78.0f,
                -33.0f, -123.0f, -21.0f, 45.0f,  0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    -72.0f, -63.0f, 27.0f,  90.0f,  54.0f,  -18.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
                0.0f,   0.0f,    0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f},
            true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_2items_plain_plain_real_unpacked) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
      CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                    46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                    -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                    -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                    48.0f,   -66.0f, -42.0f, 72.0f},
      false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_2items_plain_plain_real_packed) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
      CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                    46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                    -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                    -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                    48.0f,   -66.0f, -42.0f, 72.0f},
      false, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_2items_plain_plain_complex_unpacked) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
      CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                    46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                    -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                    -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                    48.0f,   -66.0f, -42.0f, 72.0f},
      false, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_2items_plain_plain_complex_packed) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
      CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                    46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                    -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                    -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                    48.0f,   -66.0f, -42.0f, 72.0f},
      false, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_2items_plain_cipher_real_unpacked) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
      CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                    46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                    -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                    -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                    48.0f,   -66.0f, -42.0f, 72.0f},
      false, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_2items_plain_cipher_real_packed) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
      CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                    46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                    -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                    -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                    48.0f,   -66.0f, -42.0f, 72.0f},
      false, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_2items_plain_cipher_complex_unpacked) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
      CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                    46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                    -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                    -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                    48.0f,   -66.0f, -42.0f, 72.0f},
      false, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_2items_plain_cipher_complex_packed) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
      CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                    46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                    -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                    -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                    48.0f,   -66.0f, -42.0f, 72.0f},
      false, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_2items_cipher_plain_real_unpacked) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
      CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                    46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                    -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                    -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                    48.0f,   -66.0f, -42.0f, 72.0f},
      true, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_2items_cipher_plain_real_packed) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
      CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                    46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                    -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                    -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                    48.0f,   -66.0f, -42.0f, 72.0f},
      true, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_2items_cipher_plain_complex_unpacked) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
      CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                    46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                    -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                    -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                    48.0f,   -66.0f, -42.0f, 72.0f},
      true, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_2items_cipher_plain_complex_packed) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
      CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                    46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                    -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                    -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                    48.0f,   -66.0f, -42.0f, 72.0f},
      true, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_2items_cipher_cipher_real_unpacked) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
      CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                    46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                    -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                    -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                    48.0f,   -66.0f, -42.0f, 72.0f},
      true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2d_2items_cipher_cipher_real_packed) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
      CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                    46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                    -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                    -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                    48.0f,   -66.0f, -42.0f, 72.0f},
      true, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_2items_cipher_cipher_complex_unpacked) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
      CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                    46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                    -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                    -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                    48.0f,   -66.0f, -42.0f, 72.0f},
      true, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_2items_cipher_cipher_complex_packed) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{1, 1}, Strides{1, 1},
      CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{32.0f,   -18.0f, 56.0f,  56.0f,  -42.0f, -14.0f, -16.0f,
                    46.0f,   -54.0f, -9.0f,  -30.0f, 48.0f,  78.0f,  -33.0f,
                    -123.0f, -21.0f, -52.0f, -74.0f, 82.0f,  -30.0f, -48.0f,
                    -10.0f,  8.0f,   64.0f,  138.0f, 30.0f,  -30.0f, 6.0f,
                    48.0f,   -66.0f, -42.0f, 72.0f},
      true, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_2items_strided_padded_plain_plain_real_unpacked) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{2, 2}, Strides{1, 1},
      CoordinateDiff{4, 2}, CoordinateDiff{5, 7}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  32.0f,  56.0f,  -92.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   66.0f,  0.0f,  16.0f,  0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   -54.0f, -30.0f, 81.0f, 0.0f,   0.0f,   0.0f,
          0.0f,   -63.0f, 90.0f,  -18.0f, 0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          -52.0f, 82.0f,  -28.0f, 0.0f,   0.0f,   0.0f,  0.0f,   -2.0f,  -64.0f,
          72.0f,  0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  138.0f, -30.0f, 0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   -9.0f,  27.0f, -81.0f, 0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f},
      false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_2items_strided_padded_plain_cipher_real_unpacked) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{2, 2}, Strides{1, 1},
      CoordinateDiff{4, 2}, CoordinateDiff{5, 7}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  32.0f,  56.0f,  -92.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   66.0f,  0.0f,  16.0f,  0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   -54.0f, -30.0f, 81.0f, 0.0f,   0.0f,   0.0f,
          0.0f,   -63.0f, 90.0f,  -18.0f, 0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          -52.0f, 82.0f,  -28.0f, 0.0f,   0.0f,   0.0f,  0.0f,   -2.0f,  -64.0f,
          72.0f,  0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  138.0f, -30.0f, 0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   -9.0f,  27.0f, -81.0f, 0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f},
      false, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_2items_strided_padded_cipher_plain_real_unpacked) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{2, 2}, Strides{1, 1},
      CoordinateDiff{4, 2}, CoordinateDiff{5, 7}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  32.0f,  56.0f,  -92.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   66.0f,  0.0f,  16.0f,  0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   -54.0f, -30.0f, 81.0f, 0.0f,   0.0f,   0.0f,
          0.0f,   -63.0f, 90.0f,  -18.0f, 0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          -52.0f, 82.0f,  -28.0f, 0.0f,   0.0f,   0.0f,  0.0f,   -2.0f,  -64.0f,
          72.0f,  0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  138.0f, -30.0f, 0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   -9.0f,  27.0f, -81.0f, 0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f},
      true, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            convolution_2d_2items_strided_padded_cipher_cipher_real_unpacked) {
  conv_test(
      Shape{2, 1, 3, 5}, Shape{2, 1, 2, 2}, Strides{2, 2}, Strides{1, 1},
      CoordinateDiff{4, 2}, CoordinateDiff{5, 7}, Strides{1, 1},
      vector<float>{-8.f, 2.f,  -4.f, -2.f, 9.f,  9.f,  -0.f, -3.f, -8.f, 5.f,
                    -8.f, 1.f,  2.f,  8.f,  -2.f, 6.f,  9.f,  -7.f, 3.f,  0.f,
                    6.f,  -1.f, -4.f, -2.f, 7.f,  -0.f, -1.f, 7.f,  -4.f, -9.f},
      vector<float>{-8.f, 2.f, -4.f, -2.f, 9.f, 9.f, -0.f, -3.f},
      vector<float>{
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  32.0f,  56.0f,  -92.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   66.0f,  0.0f,  16.0f,  0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   -54.0f, -30.0f, 81.0f, 0.0f,   0.0f,   0.0f,
          0.0f,   -63.0f, 90.0f,  -18.0f, 0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          -52.0f, 82.0f,  -28.0f, 0.0f,   0.0f,   0.0f,  0.0f,   -2.0f,  -64.0f,
          72.0f,  0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  138.0f, -30.0f, 0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   -9.0f,  27.0f, -81.0f, 0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,  0.0f,   0.0f,   0.0f,
          0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f},
      true, true, false, false);
}