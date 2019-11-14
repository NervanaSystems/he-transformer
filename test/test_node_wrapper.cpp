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

namespace ngraph::runtime::he {

template <typename OP>
bool check_nullary() {
  auto node = std::make_shared<OP>();
  NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  return (node_wrapper.get_node() != nullptr) &&
         (node_wrapper.get_op() != nullptr);
}

template <typename OP>
bool check_unary() {
  Shape shape{1};
  auto param = std::make_shared<op::Parameter>(element::f32, shape);
  auto node = std::make_shared<OP>(param);
  NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  return (node_wrapper.get_node() != nullptr) &&
         (node_wrapper.get_op() != nullptr);
}

template <typename OP>
void check_unsupported_unary() {
  Shape shape{1};
  auto param = std::make_shared<op::Parameter>(element::f32, shape);
  auto node = std::make_shared<OP>(param);
  NodeWrapper node_wrapper(node);

  EXPECT_ANY_THROW({ node_wrapper.get_op(); });
}

template <typename OP>
bool check_binary() {
  Shape shape{1};
  auto param = std::make_shared<op::Parameter>(element::f32, shape);
  auto node = std::make_shared<OP>(param, param);
  NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  return (node_wrapper.get_node() != nullptr) &&
         (node_wrapper.get_op() != nullptr);
}

TEST(node_wrapper, abs) { ASSERT_TRUE(check_unary<op::Abs>()); }

TEST(node_wrapper, acos) { ASSERT_TRUE(check_unary<op::Acos>()); }

TEST(node_wrapper, add) { ASSERT_TRUE(check_binary<op::Add>()); }

TEST(node_wrapper, all) { ASSERT_TRUE(check_nullary<op::All>()); }

TEST(node_wrapper, allreduce) { check_unsupported_unary<op::AllReduce>(); }

TEST(node_wrapper, argmax) { ASSERT_TRUE(check_nullary<op::ArgMax>()); }

TEST(node_wrapper, argmin) { ASSERT_TRUE(check_nullary<op::ArgMin>()); }

TEST(node_wrapper, asin) { ASSERT_TRUE(check_unary<op::Asin>()); }

TEST(node_wrapper, atan) { ASSERT_TRUE(check_unary<op::Atan>()); }

TEST(node_wrapper, avg_pool) { ASSERT_TRUE(check_nullary<op::AvgPool>()); }

TEST(node_wrapper, batch_norm_inference) {
  ASSERT_TRUE(check_nullary<op::BatchNormInference>());
}

TEST(node_wrapper, broadcast) {
  Shape shape1{1};
  auto arg0 = std::make_shared<op::Parameter>(element::f32, shape1);
  NodeVector new_args{std::make_shared<op::Parameter>(element::f32, shape1)};

  Shape shape{4, 1, 3};
  AxisSet axes{0, 2};

  auto node = std::make_shared<op::Broadcast>(arg0, shape, axes);
  NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, bounded_relu) {
  Shape shape{1};
  auto param = std::make_shared<op::Parameter>(element::f32, shape);
  auto node = std::make_shared<op::BoundedRelu>(param, 6.0f);
  NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, ceiling) { ASSERT_TRUE(check_unary<op::Ceiling>()); }

TEST(node_wrapper, concat) {
  Shape shape{1};
  auto arg0 = std::make_shared<op::Parameter>(element::f32, shape);
  auto arg1 = std::make_shared<op::Parameter>(element::f32, shape);
  size_t axis = 0;
  auto node = std::make_shared<op::Concat>(NodeVector{arg0, arg1}, axis);
  NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, constant) {
  Shape shape{};
  std::vector<float> c{2.4f};
  auto& et = element::f32;
  auto node = op::Constant::create(et, shape, c);

  NodeWrapper node_wrapper(node);

  // Remove check once Constant is an op
  EXPECT_ANY_THROW({ node_wrapper.get_op(); });
}

TEST(node_wrapper, convert) {
  Shape shape;
  auto& et = element::f64;
  auto arg0 = std::make_shared<op::Parameter>(element::f32, shape);
  auto node = std::make_shared<op::Convert>(arg0, et);
  NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, convolution) {
  ASSERT_TRUE(check_nullary<op::Convolution>());
}

TEST(node_wrapper, cos) { ASSERT_TRUE(check_unary<op::Cos>()); }

TEST(node_wrapper, cosh) { ASSERT_TRUE(check_unary<op::Cosh>()); }

TEST(node_wrapper, divide) { ASSERT_TRUE(check_binary<op::Divide>()); }

TEST(node_wrapper, dot) { ASSERT_TRUE(check_binary<op::Dot>()); }

TEST(node_wrapper, equal) { ASSERT_TRUE(check_binary<op::Equal>()); }

TEST(node_wrapper, erf) { ASSERT_TRUE(check_nullary<op::Erf>()); }

TEST(node_wrapper, exp) { ASSERT_TRUE(check_unary<op::Exp>()); }

TEST(node_wrapper, floor) { ASSERT_TRUE(check_unary<op::Floor>()); }

TEST(node_wrapper, get_output_element) {
  Shape shape{1};
  auto param = std::make_shared<op::Parameter>(element::f32, shape);
  auto node = std::make_shared<op::GetOutputElement>(param, 0);
  NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, greater_eq) { ASSERT_TRUE(check_binary<op::GreaterEq>()); }

TEST(node_wrapper, greater) { ASSERT_TRUE(check_binary<op::Greater>()); }

TEST(node_wrapper, less_eq) { ASSERT_TRUE(check_binary<op::LessEq>()); }

TEST(node_wrapper, less) { ASSERT_TRUE(check_binary<op::Less>()); }

TEST(node_wrapper, log) { ASSERT_TRUE(check_unary<op::Log>()); }

TEST(node_wrapper, max) { ASSERT_TRUE(check_nullary<op::Max>()); }

TEST(node_wrapper, maximum) { ASSERT_TRUE(check_binary<op::Maximum>()); }

TEST(node_wrapper, max_pool) { ASSERT_TRUE(check_nullary<op::MaxPool>()); }

TEST(node_wrapper, min) { ASSERT_TRUE(check_nullary<op::Min>()); }

TEST(node_wrapper, minimum) { ASSERT_TRUE(check_binary<op::Minimum>()); }

TEST(node_wrapper, multiply) { ASSERT_TRUE(check_binary<op::Multiply>()); }

TEST(node_wrapper, negative) { ASSERT_TRUE(check_unary<op::Negative>()); }

TEST(node_wrapper, not) { ASSERT_TRUE(check_unary<op::Not>()); }

TEST(node_wrapper, not_equal) { ASSERT_TRUE(check_binary<op::NotEqual>()); }

TEST(node_wrapper, one_hot) { ASSERT_TRUE(check_nullary<op::OneHot>()); }

TEST(node_wrapper, pad) {
  Shape shape{};
  auto param = std::make_shared<op::Parameter>(element::f32, shape);
  auto node = std::make_shared<op::Pad>(param, param, CoordinateDiff{},
                                        CoordinateDiff{});
  NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, parameter) {
  Shape shape{1};
  auto node = std::make_shared<op::Parameter>(element::f32, shape);
  NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, power) { ASSERT_TRUE(check_binary<op::Power>()); }

TEST(node_wrapper, product) { ASSERT_TRUE(check_binary<op::Product>()); }

TEST(node_wrapper, relu) { ASSERT_TRUE(check_unary<op::Relu>()); }

TEST(node_wrapper, reshape) { ASSERT_TRUE(check_nullary<op::Reshape>()); }

TEST(node_wrapper, result) { ASSERT_TRUE(check_nullary<op::Result>()); }

TEST(node_wrapper, reverse) {
  Shape shape{1};
  auto param = std::make_shared<op::Parameter>(element::f32, shape);
  auto node = std::make_shared<op::Reverse>(param, AxisSet{});
  NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, sigmoid) { ASSERT_TRUE(check_unary<op::Sigmoid>()); }

TEST(node_wrapper, sign) { ASSERT_TRUE(check_unary<op::Sign>()); }

TEST(node_wrapper, sin) { ASSERT_TRUE(check_unary<op::Sin>()); }

TEST(node_wrapper, sinh) { ASSERT_TRUE(check_unary<op::Sinh>()); }

TEST(node_wrapper, slice) { ASSERT_TRUE(check_nullary<op::Slice>()); }

TEST(node_wrapper, softmax) {
  Shape shape{1};
  auto param = std::make_shared<op::Parameter>(element::f32, shape);
  auto node = std::make_shared<op::Softmax>(param, AxisSet{});
  NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, sqrt) { ASSERT_TRUE(check_unary<op::Sqrt>()); }

TEST(node_wrapper, subtract) { ASSERT_TRUE(check_binary<op::Subtract>()); }

TEST(node_wrapper, sum) { ASSERT_TRUE(check_nullary<op::Sum>()); }

TEST(node_wrapper, tan) { ASSERT_TRUE(check_unary<op::Tan>()); }

TEST(node_wrapper, tanh) { ASSERT_TRUE(check_unary<op::Tanh>()); }

}  // namespace ngraph::runtime::he
