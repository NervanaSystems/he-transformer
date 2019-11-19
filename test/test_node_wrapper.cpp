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
void check_unsupported_nullary() {
  Shape shape{1};
  auto node = std::make_shared<OP>();
  NodeWrapper node_wrapper(node);

  EXPECT_ANY_THROW({ node_wrapper.get_op(); });
}

TEST(node_wrapper, abs) { ASSERT_TRUE(check_nullary<op::Abs>()); }

TEST(node_wrapper, acos) { ASSERT_TRUE(check_nullary<op::Acos>()); }

TEST(node_wrapper, add) { ASSERT_TRUE(check_nullary<op::Add>()); }

TEST(node_wrapper, all) { ASSERT_TRUE(check_nullary<op::All>()); }

TEST(node_wrapper, allreduce) { check_unsupported_nullary<op::AllReduce>(); }

TEST(node_wrapper, argmax) { ASSERT_TRUE(check_nullary<op::ArgMax>()); }

TEST(node_wrapper, argmin) { ASSERT_TRUE(check_nullary<op::ArgMin>()); }

TEST(node_wrapper, asin) { ASSERT_TRUE(check_nullary<op::Asin>()); }

TEST(node_wrapper, atan) { ASSERT_TRUE(check_nullary<op::Atan>()); }

TEST(node_wrapper, avg_pool) { ASSERT_TRUE(check_nullary<op::AvgPool>()); }

TEST(node_wrapper, batch_norm_inference) {
  ASSERT_TRUE(check_nullary<op::BatchNormInference>());
}

TEST(node_wrapper, broadcast) { ASSERT_TRUE(check_nullary<op::Broadcast>()); }

TEST(node_wrapper, bounded_relu) {
  Shape shape{1};
  auto param = std::make_shared<op::Parameter>(element::f32, shape);
  auto node = std::make_shared<op::BoundedRelu>(param, 6.0f);
  NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, ceiling) { ASSERT_TRUE(check_nullary<op::Ceiling>()); }

TEST(node_wrapper, concat) { ASSERT_TRUE(check_nullary<op::Concat>()); }

TEST(node_wrapper, constant) {
  Shape shape{};
  std::vector<float> c{2.4f};
  auto& et = element::f32;
  auto node = op::Constant::create(et, shape, c);
  NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, convert) { ASSERT_TRUE(check_nullary<op::Convert>()); }

TEST(node_wrapper, convolution) {
  ASSERT_TRUE(check_nullary<op::Convolution>());
}

TEST(node_wrapper, cos) { ASSERT_TRUE(check_nullary<op::Cos>()); }

TEST(node_wrapper, cosh) { ASSERT_TRUE(check_nullary<op::Cosh>()); }

TEST(node_wrapper, divide) { ASSERT_TRUE(check_nullary<op::Divide>()); }

TEST(node_wrapper, dot) { ASSERT_TRUE(check_nullary<op::Dot>()); }

TEST(node_wrapper, equal) { ASSERT_TRUE(check_nullary<op::Equal>()); }

TEST(node_wrapper, erf) { ASSERT_TRUE(check_nullary<op::Erf>()); }

TEST(node_wrapper, exp) { ASSERT_TRUE(check_nullary<op::Exp>()); }

TEST(node_wrapper, floor) { ASSERT_TRUE(check_nullary<op::Floor>()); }

TEST(node_wrapper, get_output_element) {
  Shape shape{1};
  auto param = std::make_shared<op::Parameter>(element::f32, shape);
  auto node = std::make_shared<op::GetOutputElement>(param, 0);
  NodeWrapper node_wrapper(node);

  EXPECT_NO_THROW(node_wrapper.get_typeid());
  ASSERT_TRUE((node_wrapper.get_node() != nullptr) &&
              (node_wrapper.get_op() != nullptr));
}

TEST(node_wrapper, greater_eq) { ASSERT_TRUE(check_nullary<op::GreaterEq>()); }

TEST(node_wrapper, greater) { ASSERT_TRUE(check_nullary<op::Greater>()); }

TEST(node_wrapper, less_eq) { ASSERT_TRUE(check_nullary<op::LessEq>()); }

TEST(node_wrapper, less) { ASSERT_TRUE(check_nullary<op::Less>()); }

TEST(node_wrapper, log) { ASSERT_TRUE(check_nullary<op::Log>()); }

TEST(node_wrapper, max) { ASSERT_TRUE(check_nullary<op::Max>()); }

TEST(node_wrapper, maximum) { ASSERT_TRUE(check_nullary<op::Maximum>()); }

TEST(node_wrapper, max_pool) { ASSERT_TRUE(check_nullary<op::MaxPool>()); }

TEST(node_wrapper, min) { ASSERT_TRUE(check_nullary<op::Min>()); }

TEST(node_wrapper, minimum) { ASSERT_TRUE(check_nullary<op::Minimum>()); }

TEST(node_wrapper, multiply) { ASSERT_TRUE(check_nullary<op::Multiply>()); }

TEST(node_wrapper, negative) { ASSERT_TRUE(check_nullary<op::Negative>()); }

TEST(node_wrapper, not) { ASSERT_TRUE(check_nullary<op::Not>()); }

TEST(node_wrapper, not_equal) { ASSERT_TRUE(check_nullary<op::NotEqual>()); }

TEST(node_wrapper, one_hot) { ASSERT_TRUE(check_nullary<op::OneHot>()); }

TEST(node_wrapper, pad) { ASSERT_TRUE(check_nullary<op::Pad>()); }

TEST(node_wrapper, parameter) { ASSERT_TRUE(check_nullary<op::Parameter>()); }

TEST(node_wrapper, power) { ASSERT_TRUE(check_nullary<op::Power>()); }

TEST(node_wrapper, product) { ASSERT_TRUE(check_nullary<op::Product>()); }

TEST(node_wrapper, relu) { ASSERT_TRUE(check_nullary<op::Relu>()); }

TEST(node_wrapper, reshape) { ASSERT_TRUE(check_nullary<op::Reshape>()); }

TEST(node_wrapper, result) { ASSERT_TRUE(check_nullary<op::Result>()); }

TEST(node_wrapper, reverse) { ASSERT_TRUE(check_nullary<op::Reverse>()); }

TEST(node_wrapper, sigmoid) { ASSERT_TRUE(check_nullary<op::Sigmoid>()); }

TEST(node_wrapper, sign) { ASSERT_TRUE(check_nullary<op::Sign>()); }

TEST(node_wrapper, sin) { ASSERT_TRUE(check_nullary<op::Sin>()); }

TEST(node_wrapper, sinh) { ASSERT_TRUE(check_nullary<op::Sinh>()); }

TEST(node_wrapper, slice) { ASSERT_TRUE(check_nullary<op::Slice>()); }

TEST(node_wrapper, softmax) { ASSERT_TRUE(check_nullary<op::Softmax>()); }

TEST(node_wrapper, sqrt) { ASSERT_TRUE(check_nullary<op::Sqrt>()); }

TEST(node_wrapper, subtract) { ASSERT_TRUE(check_nullary<op::Subtract>()); }

TEST(node_wrapper, sum) { ASSERT_TRUE(check_nullary<op::Sum>()); }

TEST(node_wrapper, tan) { ASSERT_TRUE(check_nullary<op::Tan>()); }

TEST(node_wrapper, tanh) { ASSERT_TRUE(check_nullary<op::Tanh>()); }

}  // namespace ngraph::runtime::he
