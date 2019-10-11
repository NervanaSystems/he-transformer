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

auto avg_pool_test = [](const ngraph::Shape& shape_a,
                        const ngraph::Shape& window_shape,
                        const Strides& window_movement_strides,
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
  auto t = make_shared<op::AvgPool>(a, window_shape, window_movement_strides);
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

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_1channel_1image_plain) {
  float denom = 3.0;
  avg_pool_test(
      Shape{1, 1, 14}, Shape{3}, Strides{},
      test::NDArray<float, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}
          .get_vector(),
      test::NDArray<float, 3>({{{1 / denom, 3 / denom, 3 / denom, 3 / denom,
                                 4 / denom, 5 / denom, 5 / denom, 2 / denom,
                                 2 / denom, 2 / denom, 2 / denom, 0 / denom}}})
          .get_vector(),
      false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_1channel_2image_plain) {
  float denom = 3.0;
  avg_pool_test(
      Shape{2, 1, 14}, Shape{3}, Strides{},
      test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                               {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
          .get_vector(),
      test::NDArray<float, 3>(
          {{{1 / denom, 3 / denom, 3 / denom, 3 / denom, 4 / denom, 5 / denom,
             5 / denom, 2 / denom, 2 / denom, 2 / denom, 2 / denom, 0 / denom}},
           {{3 / denom, 4 / denom, 2 / denom, 1 / denom, 0 / denom, 2 / denom,
             2 / denom, 3 / denom, 1 / denom, 1 / denom, 1 / denom,
             3 / denom}}})
          .get_vector(),
      false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_1channel_2image_cipher_real_packed) {
  float denom = 3.0;
  avg_pool_test(
      Shape{2, 1, 14}, Shape{3}, Strides{},
      test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                               {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
          .get_vector(),
      test::NDArray<float, 3>(
          {{{1 / denom, 3 / denom, 3 / denom, 3 / denom, 4 / denom, 5 / denom,
             5 / denom, 2 / denom, 2 / denom, 2 / denom, 2 / denom, 0 / denom}},
           {{3 / denom, 4 / denom, 2 / denom, 1 / denom, 0 / denom, 2 / denom,
             2 / denom, 3 / denom, 1 / denom, 1 / denom, 1 / denom,
             3 / denom}}})
          .get_vector(),
      true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_1channel_2image_plain_real_unpacked) {
  float denom = 3.0;
  avg_pool_test(
      Shape{2, 2, 14}, Shape{3}, Strides{},
      test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                               {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
          .get_vector(),
      test::NDArray<float, 3>(
          {{{1 / denom, 3 / denom, 3 / denom, 3 / denom, 4 / denom, 5 / denom,
             5 / denom, 2 / denom, 2 / denom, 2 / denom, 2 / denom, 0 / denom},
            {0 / denom, 2 / denom, 2 / denom, 2 / denom, 2 / denom, 5 / denom,
             5 / denom, 4 / denom, 3 / denom, 3 / denom, 3 / denom, 1 / denom}},

           {{3 / denom, 4 / denom, 2 / denom, 1 / denom, 0 / denom, 2 / denom,
             2 / denom, 3 / denom, 1 / denom, 1 / denom, 1 / denom, 3 / denom},
            {3 / denom, 1 / denom, 1 / denom, 1 / denom, 3 / denom, 2 / denom,
             2 / denom, 0 / denom, 1 / denom, 2 / denom, 4 / denom,
             3 / denom}}})
          .get_vector(),
      true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_plain_real_unpacked) {
  float denom = 2 * 3;
  avg_pool_test(Shape{2, 2, 5, 5}, Shape{2, 3}, Strides{},
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
                test::NDArray<float, 4>(
                    {{{{6 / denom, 8 / denom, 5 / denom},  // img 0 chan 0
                       {7 / denom, 5 / denom, 3 / denom},
                       {5 / denom, 2 / denom, 5 / denom},
                       {6 / denom, 5 / denom, 5 / denom}},

                      {{5 / denom, 7 / denom, 6 / denom},  // img 0 chan 1
                       {8 / denom, 6 / denom, 7 / denom},
                       {7 / denom, 2 / denom, 3 / denom},
                       {6 / denom, 1 / denom, 0 / denom}}},

                     {{{5 / denom, 6 / denom, 5 / denom},  // img 1 chan 0
                       {3 / denom, 5 / denom, 9 / denom},
                       {3 / denom, 6 / denom, 9 / denom},
                       {2 / denom, 3 / denom, 3 / denom}},

                      {{5 / denom, 3 / denom, 1 / denom},  // img 1 chan 1
                       {6 / denom, 5 / denom, 4 / denom},
                       {7 / denom, 5 / denom, 6 / denom},
                       {4 / denom, 2 / denom, 4 / denom}}}})
                    .get_vector(),
                false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME},
            avg_pool_2d_1channel_1image_strided_plain_real_unpacked) {
  float denom = 2 * 3;
  avg_pool_test(Shape{1, 1, 8, 8}, Shape{2, 3}, Strides{3, 2},
                test::NDArray<float, 4>({{{{0, 1, 0, 2, 1, 2, 0, 0},
                                           {0, 3, 2, 0, 0, 0, 1, 0},
                                           {2, 0, 0, 0, 1, 0, 0, 0},
                                           {2, 0, 1, 1, 2, 2, 3, 0},
                                           {0, 2, 1, 0, 0, 0, 1, 0},
                                           {2, 0, 3, 1, 0, 0, 0, 0},
                                           {1, 2, 0, 0, 0, 1, 2, 0},
                                           {1, 0, 2, 0, 0, 0, 1, 0}}}})
                    .get_vector(),
                test::NDArray<float, 4>({{{{6 / denom, 5 / denom, 4 / denom},
                                           {6 / denom, 5 / denom, 8 / denom},
                                           {6 / denom, 2 / denom, 4 / denom}}}})
                    .get_vector(),
                false, false, false);
}
