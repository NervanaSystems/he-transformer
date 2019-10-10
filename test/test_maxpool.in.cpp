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

auto max_pool_test = [](const ngraph::Shape& shape_a,
                        const ngraph::Shape& window_shape,
                        const vector<float>& input_a,
                        const vector<float>& output, const bool arg1_encrypted,
                        const bool complex_packing, const bool packed) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        ngraph::he::HESealEncryptionParameters::
            default_complex_packing_parms());
  }

  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto t = make_shared<op::MaxPool>(a, window_shape);
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

  copy_data(t_a, input_a);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a});
  EXPECT_TRUE(all_close(read_vector<float>(t_result), output, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_1channel_1image_plain_real_unpacked) {
  max_pool_test(Shape{1, 1, 14}, Shape{3},
                vector<float>{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                vector<float>{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}, false, false,
                false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_1channel_1image_cipher_real_unpacked) {
  max_pool_test(Shape{1, 1, 14}, Shape{3},
                vector<float>{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                vector<float>{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}, true, false,
                false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_1channel_2image_plain_real_unpacked) {
  max_pool_test(
      Shape{2, 1, 14}, Shape{3},
      test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                               {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
          .get_vector(),
      test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}},
                               {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
          .get_vector(),
      false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_1channel_2image_plain_real_packed) {
  max_pool_test(
      Shape{2, 1, 14}, Shape{3},
      test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                               {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
          .get_vector(),
      test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}},
                               {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
          .get_vector(),
      false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_1channel_2image_cipher_real_unpacked) {
  max_pool_test(
      Shape{2, 1, 14}, Shape{3},
      test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                               {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
          .get_vector(),
      test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}},
                               {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
          .get_vector(),
      true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_1channel_2image_cipher_real_packed) {
  max_pool_test(
      Shape{2, 1, 14}, Shape{3},
      test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                               {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
          .get_vector(),
      test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}},
                               {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
          .get_vector(),
      true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_2channel_2image_plain_real_unpacked) {
  max_pool_test(
      Shape{2, 2, 14}, Shape{3},
      test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                               {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
          .get_vector(),
      test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0},
                                {0, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1}},

                               {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2},
                                {2, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2}}})
          .get_vector(),
      false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_2channel_2image_cipher_real_unpacked) {
  max_pool_test(
      Shape{2, 2, 14}, Shape{3},
      test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                               {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
          .get_vector(),
      test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0},
                                {0, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1}},

                               {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2},
                                {2, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2}}})
          .get_vector(),
      true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_2d_2channel_2image_plain_real_unpacked) {
  max_pool_test(Shape{2, 2, 5, 5}, Shape{2, 3},
                test::NDArray<float, 4>({{{{0, 1, 0, 2, 1},  // img 0 chan 0
                                           {0, 3, 2, 0, 0},
                                           {2, 0, 0, 0, 1},
                                           {2, 0, 1, 1, 2},
                                           {0, 2, 1, 0, 0}},

                                          {{0, 0, 0, 2, 0},  // img 0 chan 1
                                           {0, 2, 3, 0, 1},
                                           {2, 0, 1, 0, 2},
                                           {3, 1, 0, 0, 0},
                                           {2, 0, 0, 0, 0}}},

                                         {{{0, 2, 1, 1, 0},  // img 1 chan 0
                                           {0, 0, 2, 0, 1},
                                           {0, 0, 1, 2, 3},
                                           {2, 0, 0, 3, 0},
                                           {0, 0, 0, 0, 0}},

                                          {{2, 1, 0, 0, 1},  // img 1 chan 1
                                           {0, 2, 0, 0, 0},
                                           {1, 1, 2, 0, 2},
                                           {1, 1, 1, 0, 1},
                                           {1, 0, 0, 0, 2}}}})
                    .get_vector(),
                test::NDArray<float, 4>({{{{3, 3, 2},  // img 0 chan 0
                                           {3, 3, 2},
                                           {2, 1, 2},
                                           {2, 2, 2}},

                                          {{3, 3, 3},  // img 0 chan 1
                                           {3, 3, 3},
                                           {3, 1, 2},
                                           {3, 1, 0}}},

                                         {{{2, 2, 2},  // img 1 chan 0
                                           {2, 2, 3},
                                           {2, 3, 3},
                                           {2, 3, 3}},

                                          {{2, 2, 1},  // img 1 chan 1
                                           {2, 2, 2},
                                           {2, 2, 2},
                                           {1, 1, 2}}}})
                    .get_vector(),
                false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_2d_2channel_2image_cipher_real_unpacked) {
  max_pool_test(Shape{2, 2, 5, 5}, Shape{2, 3},
                test::NDArray<float, 4>({{{{0, 1, 0, 2, 1},  // img 0 chan 0
                                           {0, 3, 2, 0, 0},
                                           {2, 0, 0, 0, 1},
                                           {2, 0, 1, 1, 2},
                                           {0, 2, 1, 0, 0}},

                                          {{0, 0, 0, 2, 0},  // img 0 chan 1
                                           {0, 2, 3, 0, 1},
                                           {2, 0, 1, 0, 2},
                                           {3, 1, 0, 0, 0},
                                           {2, 0, 0, 0, 0}}},

                                         {{{0, 2, 1, 1, 0},  // img 1 chan 0
                                           {0, 0, 2, 0, 1},
                                           {0, 0, 1, 2, 3},
                                           {2, 0, 0, 3, 0},
                                           {0, 0, 0, 0, 0}},

                                          {{2, 1, 0, 0, 1},  // img 1 chan 1
                                           {0, 2, 0, 0, 0},
                                           {1, 1, 2, 0, 2},
                                           {1, 1, 1, 0, 1},
                                           {1, 0, 0, 0, 2}}}})
                    .get_vector(),
                test::NDArray<float, 4>({{{{3, 3, 2},  // img 0 chan 0
                                           {3, 3, 2},
                                           {2, 1, 2},
                                           {2, 2, 2}},

                                          {{3, 3, 3},  // img 0 chan 1
                                           {3, 3, 3},
                                           {3, 1, 2},
                                           {3, 1, 0}}},

                                         {{{2, 2, 2},  // img 1 chan 0
                                           {2, 2, 3},
                                           {2, 3, 3},
                                           {2, 3, 3}},

                                          {{2, 2, 1},  // img 1 chan 1
                                           {2, 2, 2},
                                           {2, 2, 2},
                                           {1, 1, 2}}}})
                    .get_vector(),
                false, false, false);
}