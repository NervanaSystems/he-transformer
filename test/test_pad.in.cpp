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
#include "ngraph/op/pad.hpp"
#include "seal/he_seal_backend.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

static std::string s_manifest = "${MANIFEST}";

namespace ngraph::he {

auto pad_test = [](const Shape& shape_a, const CoordinateDiff& padding_below,
                   const CoordinateDiff& padding_above,
                   const op::PadMode& pad_mode,
                   const std::vector<float>& input_a,
                   const std::vector<float>& input_b,
                   const std::vector<float>& output, const bool arg1_encrypted,
                   const bool arg2_encrypted, const bool complex_packing,
                   const bool packed) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        HESealEncryptionParameters::default_complex_packing_parms());
  }

  NGRAPH_INFO << "padding_below " << padding_below;
  NGRAPH_INFO << "padding_above " << padding_above;
  NGRAPH_INFO << "shape_a " << shape_a;

  auto a = std::make_shared<op::Parameter>(element::f32, shape_a);
  auto b = std::make_shared<op::Parameter>(element::f32, Shape{});
  auto t =
      std::make_shared<op::Pad>(a, b, padding_below, padding_above, pad_mode);
  auto f = std::make_shared<Function>(t, ParameterVector{a, b});

  const auto& arg1_config =
      test::config_from_flags(false, arg1_encrypted, packed);
  const auto& arg2_config =
      test::config_from_flags(false, arg2_encrypted, packed);

  std::string error_str;
  he_backend->set_config(
      {{a->get_name(), arg1_config}, {b->get_name(), arg2_config}}, error_str);

  auto t_a =
      test::tensor_from_flags(*he_backend, shape_a, arg1_encrypted, packed);
  auto t_b =
      test::tensor_from_flags(*he_backend, Shape{}, arg2_encrypted, packed);
  auto t_result = test::tensor_from_flags(
      *he_backend, t->get_shape(), arg1_encrypted || arg2_encrypted, packed);

  copy_data(t_a, input_a);
  copy_data(t_b, input_b);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a, t_b});
  EXPECT_TRUE(test::all_close(read_vector<float>(t_result), output, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_1d_plain) {
  pad_test(Shape{6}, CoordinateDiff{4}, CoordinateDiff{5},
           op::PadMode::CONSTANT, std::vector<float>{{1, 2, 3, 4, 5, 6}},
           std::vector<float>{2112},
           std::vector<float>{2112, 2112, 2112, 2112, 1, 2, 3, 4, 5, 6, 2112,
                              2112, 2112, 2112, 2112},
           false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_1d_cipher) {
  pad_test(Shape{6}, CoordinateDiff{4}, CoordinateDiff{5},
           op::PadMode::CONSTANT, std::vector<float>{{1, 2, 3, 4, 5, 6}},
           std::vector<float>{2112},
           std::vector<float>{2112, 2112, 2112, 2112, 1, 2, 3, 4, 5, 6, 2112,
                              2112, 2112, 2112, 2112},
           true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_1d) {
  pad_test(Shape{6}, CoordinateDiff{4}, CoordinateDiff{-2},
           op::PadMode::CONSTANT, std::vector<float>{{1, 2, 3, 4, 5, 6}},
           std::vector<float>{2112},
           std::vector<float>{2112, 2112, 2112, 2112, 1, 2, 3, 4}, true, true,
           false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_1d_check_limits) {
  pad_test(Shape{6}, CoordinateDiff{4}, CoordinateDiff{-7},
           op::PadMode::CONSTANT, std::vector<float>{{1, 2, 3, 4, 5, 6}},
           std::vector<float>{2112}, std::vector<float>{2112, 2112, 2112}, true,
           true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d) {
  pad_test(Shape{6}, CoordinateDiff{2}, CoordinateDiff{3}, op::PadMode::EDGE,
           std::vector<float>{1, 2, 3, 4, 5, 6}, std::vector<float>{2112},
           std::vector<float>{1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6}, true, true,
           false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d_top_neg) {
  pad_test(Shape{6}, CoordinateDiff{2}, CoordinateDiff{-3}, op::PadMode::EDGE,
           std::vector<float>{{1, 2, 3, 4, 5, 6}}, std::vector<float>{2112},
           std::vector<float>{1, 1, 1, 2, 3}, true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d_top_neg_bigger_than_tensor) {
  pad_test(Shape{6}, CoordinateDiff{2}, CoordinateDiff{-7}, op::PadMode::EDGE,
           std::vector<float>{{1, 2, 3, 4, 5, 6}}, std::vector<float>{2112},
           std::vector<float>{1}, true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d_bottom_neg) {
  pad_test(Shape{6}, CoordinateDiff{-2}, CoordinateDiff{3}, op::PadMode::EDGE,
           std::vector<float>{{1, 2, 3, 4, 5, 6}}, std::vector<float>{2112},
           std::vector<float>{3, 4, 5, 6, 6, 6, 6}, true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d_bottom_neg_bigger_than_tensor) {
  pad_test(Shape{6}, CoordinateDiff{-7}, CoordinateDiff{3}, op::PadMode::EDGE,
           std::vector<float>{{1, 2, 3, 4, 5, 6}}, std::vector<float>{2112},
           std::vector<float>{6, 6}, true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_2d) {
  pad_test(Shape{3, 4}, CoordinateDiff{2, 3}, CoordinateDiff{1, 2},
           op::PadMode::EDGE,
           std::vector<float>{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
           std::vector<float>{2112},
           ngraph::test::NDArray<float, 2>({{1, 1, 1, 1, 2, 3, 4, 4, 4},
                                    {1, 1, 1, 1, 2, 3, 4, 4, 4},
                                    {1, 1, 1, 1, 2, 3, 4, 4, 4},
                                    {5, 5, 5, 5, 6, 7, 8, 8, 8},
                                    {9, 9, 9, 9, 10, 11, 12, 12, 12},
                                    {9, 9, 9, 9, 10, 11, 12, 12, 12}})
               .get_vector(),
           true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_2d_with_neg) {
  pad_test(
      Shape{3, 4}, CoordinateDiff{2, -1}, CoordinateDiff{1, 2},
      op::PadMode::EDGE,
      ngraph::test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}})
          .get_vector(),
      std::vector<float>{2112},
      ngraph::test::NDArray<float, 2>({{2, 3, 4, 4, 4},
                               {2, 3, 4, 4, 4},
                               {2, 3, 4, 4, 4},
                               {6, 7, 8, 8, 8},
                               {10, 11, 12, 12, 12},
                               {10, 11, 12, 12, 12}})
          .get_vector(),
      true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d) {
  pad_test(
      Shape{6}, CoordinateDiff{2}, CoordinateDiff{3}, op::PadMode::REFLECT,
      std::vector<float>{1, 2, 3, 4, 5, 6}, std::vector<float>{2112},
      ngraph::test::NDArray<float, 1>({3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3}).get_vector(),
      true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_top_neg) {
  pad_test(Shape{6}, CoordinateDiff{2}, CoordinateDiff{-3},
           op::PadMode::REFLECT, std::vector<float>{1, 2, 3, 4, 5, 6},
           std::vector<float>{2112},
           ngraph::test::NDArray<float, 1>({3, 2, 1, 2, 3}).get_vector(), true, true,
           false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_top_neg_bigger_than_tensor) {
  pad_test(Shape{6}, CoordinateDiff{2}, CoordinateDiff{-7},
           op::PadMode::REFLECT, std::vector<float>{1, 2, 3, 4, 5, 6},
           std::vector<float>{2112}, ngraph::test::NDArray<float, 1>({3}).get_vector(),
           true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_bottom_neg) {
  pad_test(Shape{6}, CoordinateDiff{-2}, CoordinateDiff{3},
           op::PadMode::REFLECT, std::vector<float>{1, 2, 3, 4, 5, 6},
           std::vector<float>{2112},
           ngraph::test::NDArray<float, 1>({3, 4, 5, 6, 5, 4, 3}).get_vector(), true,
           true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_bottom_neg_bigger_than_tensor) {
  pad_test(
      Shape{6}, CoordinateDiff{-7}, CoordinateDiff{3}, op::PadMode::REFLECT,
      std::vector<float>{1, 2, 3, 4, 5, 6}, std::vector<float>{2112},
      ngraph::test::NDArray<float, 1>({4, 3}).get_vector(), true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_multi_reflect) {
  pad_test(Shape{3}, CoordinateDiff{10}, CoordinateDiff{9},
           op::PadMode::REFLECT, std::vector<float>{1, 2, 3},
           std::vector<float>{2112},
           ngraph::test::NDArray<float, 1>({3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1,
                                    2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2})
               .get_vector(),
           true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_2d) {
  pad_test(
      Shape{3, 4}, CoordinateDiff{2, 3}, CoordinateDiff{1, 2},
      op::PadMode::REFLECT,
      ngraph::test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}})
          .get_vector(),
      std::vector<float>{2112},
      ngraph::test::NDArray<float, 2>({{12, 11, 10, 9, 10, 11, 12, 11, 10},
                               {8, 7, 6, 5, 6, 7, 8, 7, 6},
                               {4, 3, 2, 1, 2, 3, 4, 3, 2},
                               {8, 7, 6, 5, 6, 7, 8, 7, 6},
                               {12, 11, 10, 9, 10, 11, 12, 11, 10},
                               {8, 7, 6, 5, 6, 7, 8, 7, 6}})
          .get_vector(),
      true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_2d_with_neg) {
  pad_test(
      Shape{3, 4}, CoordinateDiff{2, -1}, CoordinateDiff{1, 2},
      op::PadMode::REFLECT,
      ngraph::test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}})
          .get_vector(),
      std::vector<float>{2112},
      ngraph::test::NDArray<float, 2>({{10, 11, 12, 11, 10},
                               {6, 7, 8, 7, 6},
                               {2, 3, 4, 3, 2},
                               {6, 7, 8, 7, 6},
                               {10, 11, 12, 11, 10},
                               {6, 7, 8, 7, 6}})
          .get_vector(),
      true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_2d) {
  pad_test(Shape{2, 3}, CoordinateDiff{1, -1}, CoordinateDiff{2, 0},
           op::PadMode::CONSTANT,
           ngraph::test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector(),
           std::vector<float>{9},
           ngraph::test::NDArray<float, 2>({{9, 9}, {2, 3}, {5, 6}, {9, 9}, {9, 9}})
               .get_vector(),
           true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_2d_all_negative) {
  pad_test(
      Shape{3, 3}, CoordinateDiff{-1, -1}, CoordinateDiff{-1, -1},
      op::PadMode::CONSTANT,
      ngraph::test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}).get_vector(),
      std::vector<float>{9}, ngraph::test::NDArray<float, 2>({{5}}).get_vector(), true,
      true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_0x0) {
  pad_test(Shape{0, 0}, CoordinateDiff{2, 3}, CoordinateDiff{3, 2},
           op::PadMode::CONSTANT, std::vector<float>{},
           std::vector<float>{2112},
           ngraph::test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                    {2112, 2112, 2112, 2112, 2112},
                                    {2112, 2112, 2112, 2112, 2112},
                                    {2112, 2112, 2112, 2112, 2112},
                                    {2112, 2112, 2112, 2112, 2112}})
               .get_vector(),
           true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_0x3) {
  pad_test(Shape{0, 3}, CoordinateDiff{2, 1}, CoordinateDiff{3, 1},
           op::PadMode::CONSTANT, std::vector<float>{},
           std::vector<float>{2112},
           ngraph::test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                    {2112, 2112, 2112, 2112, 2112},
                                    {2112, 2112, 2112, 2112, 2112},
                                    {2112, 2112, 2112, 2112, 2112},
                                    {2112, 2112, 2112, 2112, 2112}})
               .get_vector(),
           true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_3x0) {
  pad_test(Shape{3, 0}, CoordinateDiff{1, 3}, CoordinateDiff{1, 2},
           op::PadMode::CONSTANT, std::vector<float>{},
           std::vector<float>{2112},
           ngraph::test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                    {2112, 2112, 2112, 2112, 2112},
                                    {2112, 2112, 2112, 2112, 2112},
                                    {2112, 2112, 2112, 2112, 2112},
                                    {2112, 2112, 2112, 2112, 2112}})
               .get_vector(),
           true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_4d_1x2x2x2) {
  pad_test(Shape{1, 2, 2, 2}, CoordinateDiff{0, 0, 1, 1},
           CoordinateDiff{0, 0, 1, 1}, op::PadMode::CONSTANT,
           ngraph::test::NDArray<float, 4>(
               {{{{0.0f, 0.0f}, {0.0f, 0.0f}}, {{0.0f, 0.0f}, {0.0f, 0.0f}}}})
               .get_vector(),
           std::vector<float>{42},
           ngraph::test::NDArray<float, 4>({{{{42.0f, 42.0f, 42.0f, 42.0f},
                                      {42.0f, 0.0f, 0.0f, 42.0f},
                                      {42.0f, 0.0f, 0.0f, 42.0f},
                                      {42.0f, 42.0f, 42.0f, 42.0f}},
                                     {{42.0f, 42.0f, 42.0f, 42.0f},
                                      {42.0f, 0.0f, 0.0f, 42.0f},
                                      {42.0f, 0.0f, 0.0f, 42.0f},
                                      {42.0f, 42.0f, 42.0f, 42.0f}}}})
               .get_vector(),
           true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_4d) {
  pad_test(Shape{1, 3, 2, 2}, CoordinateDiff{0, -1, 1, 1},
           CoordinateDiff{0, -1, 1, 1}, op::PadMode::CONSTANT,
           ngraph::test::NDArray<float, 4>({{{{0.0f, 0.0f}, {0.0f, 0.0f}},
                                     {{1.0f, 1.0f}, {1.0f, 1.0f}},
                                     {{2.0f, 2.0f}, {2.0f, 2.0f}}}})
               .get_vector(),
           std::vector<float>{42},
           ngraph::test::NDArray<float, 4>({{{{42.0f, 42.0f, 42.0f, 42.0f},
                                      {42.0f, 1.0f, 1.0f, 42.0f},
                                      {42.0f, 1.0f, 1.0f, 42.0f},
                                      {42.0f, 42.0f, 42.0f, 42.0f}}}})
               .get_vector(),
           true, true, false, false);
}

// This test covers the case with multiple image and with asymetric pad
// bug has been found on nvGPU side now covered by this test
NGRAPH_TEST(${BACKEND_NAME}, pad_2channel_2image_asym) {
  pad_test(Shape{2, 2, 4, 4}, CoordinateDiff{0, 0, 0, 0},
           CoordinateDiff{0, 0, 2, 2}, op::PadMode::CONSTANT,
           ngraph::test::NDArray<float, 4>({{{{0, 1, 0, 2},  // img 0 chan 0
                                      {0, 3, 2, 0},
                                      {2, 0, 0, 0},
                                      {0, 2, 1, 0}},

                                     {{0, 0, 0, 2},  // img 0 chan 1
                                      {0, 2, 3, 0},
                                      {2, 0, 1, 0},
                                      {2, 0, 0, 0}}},

                                    {{{0, 2, 1, 1},  // img 1 chan 0
                                      {0, 0, 2, 0},
                                      {0, 0, 1, 2},
                                      {0, 0, 0, 0}},

                                     {{2, 1, 0, 0},  // img 1 chan 1
                                      {0, 2, 0, 0},
                                      {1, 1, 2, 0},
                                      {1, 0, 0, 0}}}})
               .get_vector(),
           std::vector<float>{42},
           ngraph::test::NDArray<float, 4>({{{{0, 1, 0, 2, 42, 42},  // img 0 chan 0
                                      {0, 3, 2, 0, 42, 42},
                                      {2, 0, 0, 0, 42, 42},
                                      {0, 2, 1, 0, 42, 42},
                                      {42, 42, 42, 42, 42, 42},
                                      {42, 42, 42, 42, 42, 42}},

                                     {{0, 0, 0, 2, 42, 42},  // img 1 chan 0
                                      {0, 2, 3, 0, 42, 42},
                                      {2, 0, 1, 0, 42, 42},
                                      {2, 0, 0, 0, 42, 42},
                                      {42, 42, 42, 42, 42, 42},
                                      {42, 42, 42, 42, 42, 42}}},

                                    {{{0, 2, 1, 1, 42, 42},  // img 1 chan 0
                                      {0, 0, 2, 0, 42, 42},
                                      {0, 0, 1, 2, 42, 42},
                                      {0, 0, 0, 0, 42, 42},
                                      {42, 42, 42, 42, 42, 42},
                                      {42, 42, 42, 42, 42, 42}},

                                     {{2, 1, 0, 0, 42, 42},  // img 1 chan 1
                                      {0, 2, 0, 0, 42, 42},
                                      {1, 1, 2, 0, 42, 42},
                                      {1, 0, 0, 0, 42, 42},
                                      {42, 42, 42, 42, 42, 42},
                                      {42, 42, 42, 42, 42, 42}}}})
               .get_vector(),
           true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_symmetric) {
  // Symmetric mode padding not supported
  EXPECT_ANY_THROW(
      pad_test(Shape{2, 3}, CoordinateDiff{1, 2}, CoordinateDiff{1, 2},
               op::PadMode::SYMMETRIC,
               ngraph::test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector(),
               std::vector<float>{2112},
               ngraph::test::NDArray<float, 2>({{2, 1, 1, 2, 3, 3, 2},
                                        {2, 1, 1, 2, 3, 3, 2},
                                        {5, 4, 4, 5, 6, 6, 5},
                                        {5, 4, 4, 5, 6, 6, 5}})
                   .get_vector(),
               true, false, false, false));
}

}  // namespace ngraph::he
