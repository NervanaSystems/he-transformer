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

#include <memory>

#include "gtest/gtest.h"
#include "he_tensor.hpp"
#include "he_type.hpp"
#include "ngraph/ngraph.hpp"
#include "op/bounded_relu.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_executable.hpp"
#include "test_util.hpp"
#include "util/test_tools.hpp"

template <typename OP>
bool check_unary() {}

TEST(bounded_relu, copy) {
  ngraph::Shape shape{1};
  auto arg0 =
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  ngraph::NodeVector new_args{
      std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape)};

  auto node = std::make_shared<ngraph::op::BoundedRelu>(arg0, 7.2);
  auto new_node = node->copy_with_new_args(new_args);

  EXPECT_TRUE(new_node != nullptr);
  EXPECT_TRUE(new_args == new_node->get_arguments());

  auto new_bounded_relu =
      std::static_pointer_cast<ngraph::op::BoundedRelu>(new_node);
  EXPECT_TRUE(new_bounded_relu->get_alpha() == node->get_alpha());
}
