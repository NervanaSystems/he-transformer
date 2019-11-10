//*****************************************************************************
// node_wrapperright 2018-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a node_wrapper of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <complex>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "he_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/passthrough.hpp"
#include "ngraph/type/element_type.hpp"
#include "node_wrapper.hpp"
#include "op/bounded_relu.hpp"
#include "test_util.hpp"
#include "util/test_tools.hpp"

template <typename OP>
bool check_nullary() {
  auto node = std::make_shared<OP>();
  ngraph::he::NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  return (node_wrapper.get_node() != nullptr) &&
         (node_wrapper.get_op() != nullptr);
}

template <typename OP>
bool check_unary() {
  ngraph::Shape shape{1};
  auto param =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto node = std::make_shared<OP>(param);
  ngraph::he::NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  return (node_wrapper.get_node() != nullptr) &&
         (node_wrapper.get_op() != nullptr);
}

template <typename OP>
bool check_binary() {
  ngraph::Shape shape{1};
  auto param =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto node = std::make_shared<OP>(param, param);
  ngraph::he::NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  return (node_wrapper.get_node() != nullptr) &&
         (node_wrapper.get_op() != nullptr);
}

TEST(node_wrapper, abs) { ASSERT_TRUE(check_unary<ngraph::op::Abs>()); }

TEST(node_wrapper, acos) { ASSERT_TRUE(check_unary<ngraph::op::Acos>()); }

TEST(node_wrapper, add) { ASSERT_TRUE(check_binary<ngraph::op::Add>()); }

TEST(node_wrapper, all) { ASSERT_TRUE(check_nullary<ngraph::op::All>()); }

TEST(node_wrapper, argmax) { ASSERT_TRUE(check_nullary<ngraph::op::ArgMax>()); }

TEST(node_wrapper, argmin) { ASSERT_TRUE(check_nullary<ngraph::op::ArgMin>()); }

TEST(node_wrapper, asin) { ASSERT_TRUE(check_unary<ngraph::op::Asin>()); }

TEST(node_wrapper, atan) { ASSERT_TRUE(check_unary<ngraph::op::Atan>()); }

TEST(node_wrapper, avg_pool) {
  ASSERT_TRUE(check_nullary<ngraph::op::AvgPool>());
}

TEST(node_wrapper, batch_norm_inference) {
  ASSERT_TRUE(check_nullary<ngraph::op::BatchNormInference>());
}

TEST(node_wrapper, broadcast) {
  ngraph::Shape shape1{1};
  auto arg0 =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape1);
  ngraph::NodeVector new_args{
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape1)};

  ngraph::Shape shape{4, 1, 3};
  ngraph::AxisSet axes{0, 2};

  auto node = std::make_shared<ngraph::op::Broadcast>(arg0, shape, axes);
  ngraph::he::NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, bounded_relu) {
  ngraph::Shape shape{1};
  auto param =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto node = std::make_shared<ngraph::op::BoundedRelu>(param, 6.0f);
  ngraph::he::NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, ceiling) { ASSERT_TRUE(check_unary<ngraph::op::Ceiling>()); }

TEST(node_wrapper, concat) {
  ngraph::Shape shape{1};
  auto arg0 =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto arg1 =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  size_t axis = 0;
  auto node = std::make_shared<ngraph::op::Concat>(
      ngraph::NodeVector{arg0, arg1}, axis);
  ngraph::he::NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, constant) {
  ngraph::Shape shape{};
  std::vector<float> c{2.4f};
  auto& et = ngraph::element::f32;
  auto node = ngraph::op::Constant::create(et, shape, c);

  ngraph::he::NodeWrapper node_wrapper(node);

  // Remove check once Constant is an op
  EXPECT_ANY_THROW({ node_wrapper.get_op(); });
}

TEST(node_wrapper, convert) {
  ngraph::Shape shape;
  auto& et = ngraph::element::f64;
  auto arg0 =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto node = std::make_shared<ngraph::op::Convert>(arg0, et);
  ngraph::he::NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, convolution) {
  ASSERT_TRUE(check_nullary<ngraph::op::Convolution>());
}

TEST(node_wrapper, cos) { ASSERT_TRUE(check_unary<ngraph::op::Cos>()); }

TEST(node_wrapper, cosh) { ASSERT_TRUE(check_unary<ngraph::op::Cosh>()); }

TEST(node_wrapper, divide) { ASSERT_TRUE(check_binary<ngraph::op::Divide>()); }

TEST(node_wrapper, dot) { ASSERT_TRUE(check_binary<ngraph::op::Dot>()); }

TEST(node_wrapper, equal) { ASSERT_TRUE(check_binary<ngraph::op::Equal>()); }

TEST(node_wrapper, erf) { ASSERT_TRUE(check_nullary<ngraph::op::Erf>()); }

TEST(node_wrapper, exp) { ASSERT_TRUE(check_unary<ngraph::op::Exp>()); }

TEST(node_wrapper, floor) { ASSERT_TRUE(check_unary<ngraph::op::Floor>()); }

TEST(node_wrapper, get_output_element) {
  ngraph::Shape shape{1};
  auto param =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto node = std::make_shared<ngraph::op::GetOutputElement>(param, 0);
  ngraph::he::NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, greater_eq) {
  ASSERT_TRUE(check_binary<ngraph::op::GreaterEq>());
}

TEST(node_wrapper, greater) {
  ASSERT_TRUE(check_binary<ngraph::op::Greater>());
}

TEST(node_wrapper, less_eq) { ASSERT_TRUE(check_binary<ngraph::op::LessEq>()); }

TEST(node_wrapper, less) { ASSERT_TRUE(check_binary<ngraph::op::Less>()); }

TEST(node_wrapper, log) { ASSERT_TRUE(check_unary<ngraph::op::Log>()); }

TEST(node_wrapper, max) { ASSERT_TRUE(check_nullary<ngraph::op::Max>()); }

TEST(node_wrapper, maximum) {
  ASSERT_TRUE(check_binary<ngraph::op::Maximum>());
}

TEST(node_wrapper, max_pool) {
  ASSERT_TRUE(check_nullary<ngraph::op::MaxPool>());
}

TEST(node_wrapper, min) { ASSERT_TRUE(check_nullary<ngraph::op::Min>()); }

TEST(node_wrapper, minimum) {
  ASSERT_TRUE(check_binary<ngraph::op::Minimum>());
}

TEST(node_wrapper, multiply) {
  ASSERT_TRUE(check_binary<ngraph::op::Multiply>());
}

TEST(node_wrapper, negative) {
  ASSERT_TRUE(check_unary<ngraph::op::Negative>());
}

TEST(node_wrapper, not_equal) {
  ASSERT_TRUE(check_binary<ngraph::op::NotEqual>());
}

TEST(node_wrapper, one_hot) {
  ASSERT_TRUE(check_nullary<ngraph::op::OneHot>());
}

TEST(node_wrapper, pad) {
  ngraph::Shape shape{};
  auto param =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto node = std::make_shared<ngraph::op::Pad>(
      param, param, ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{});
  ngraph::he::NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, parameter) {
  ngraph::Shape shape{1};
  auto node =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  ngraph::he::NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, power) { ASSERT_TRUE(check_binary<ngraph::op::Power>()); }

TEST(node_wrapper, reshape) {
  ASSERT_TRUE(check_nullary<ngraph::op::Reshape>());
}

TEST(node_wrapper, result) { ASSERT_TRUE(check_nullary<ngraph::op::Result>()); }

TEST(node_wrapper, reverse) {
  ngraph::Shape shape{1};
  auto param =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto node = std::make_shared<ngraph::op::Reverse>(param, ngraph::AxisSet{});
  ngraph::he::NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, sigmoid) { ASSERT_TRUE(check_unary<ngraph::op::Sigmoid>()); }

TEST(node_wrapper, sign) { ASSERT_TRUE(check_unary<ngraph::op::Sign>()); }

TEST(node_wrapper, sin) { ASSERT_TRUE(check_unary<ngraph::op::Sin>()); }

TEST(node_wrapper, sinh) { ASSERT_TRUE(check_unary<ngraph::op::Sinh>()); }

TEST(node_wrapper, slice) { ASSERT_TRUE(check_nullary<ngraph::op::Slice>()); }

TEST(node_wrapper, softmax) {
  ngraph::Shape shape{1};
  auto param =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto node = std::make_shared<ngraph::op::Softmax>(param, ngraph::AxisSet{});
  ngraph::he::NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, sqrt) { ASSERT_TRUE(check_unary<ngraph::op::Sqrt>()); }

TEST(node_wrapper, subtract) {
  ASSERT_TRUE(check_binary<ngraph::op::Subtract>());
}

TEST(node_wrapper, sum) { ASSERT_TRUE(check_nullary<ngraph::op::Sum>()); }

TEST(node_wrapper, tan) { ASSERT_TRUE(check_unary<ngraph::op::Tan>()); }

TEST(node_wrapper, tanh) { ASSERT_TRUE(check_unary<ngraph::op::Tanh>()); }
