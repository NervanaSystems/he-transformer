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

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "pass/he_fusion.hpp"

void ngraph::he::pass::HEFusion::construct_bounded_relu() {
  auto relu_input = std::make_shared<pattern::op::Label>(element::f32, Shape{});
  auto relu = std::make_shared<ngraph::op::Relu>(relu_input);
  auto iconst1 = ngraph::op::Constant::create(element::f32, Shape{}, {1});
  auto alpha = std::make_shared<pattern::op::Label>(iconst1);
  auto broadcast_pred = [](std::shared_ptr<Node> n) {
    return (std::dynamic_pointer_cast<ngraph::op::Broadcast>(n) != nullptr);
  };
  NGRAPH_INFO << "starting Boudned relu pass";
  auto skip_broadcast =
      std::make_shared<pattern::op::Skip>(alpha, broadcast_pred);
  auto min = std::make_shared<ngraph::op::Minimum>(relu, skip_broadcast);

  auto callback = [relu_input, alpha](pattern::Matcher& m) {
    NGRAPH_DEBUG << "In a callback for construct_bounded_relu against "
                 << m.get_match_root()->get_name();

    if (m.get_match_root()->get_element_type() != element::f32) {
      NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                   << " type is not float!";
      return false;
    }
    auto pattern_map = m.get_pattern_map();
    if (!std::dynamic_pointer_cast<ngraph::op::Constant>(pattern_map[alpha])) {
      NGRAPH_DEBUG << "alpha must be constant for bounded relu";
      return false;
    }

    // we wont fuse if the alpha and the Relu output element type are not same
    if (pattern_map[alpha]->get_element_type() !=
        pattern_map[relu_input]->get_element_type()) {
      return false;
    }
    if (pattern_map[alpha]->get_shape() !=
        pattern_map[relu_input]->get_shape()) {
      return false;
    }

    auto alpha_const_op =
        std::static_pointer_cast<ngraph::op::Constant>(pattern_map[alpha]);
    float alpha_val =
        *(static_cast<float const*>(alpha_const_op->get_data_ptr()));
    NGRAPH_DEBUG << "relu_input: " << pattern_map[relu_input] << " min_val: "
                 << *(static_cast<float const*>(
                        alpha_const_op->get_data_ptr()));

    auto cg = std::shared_ptr<Node>(
        new ngraph::op::BoundedRelu(pattern_map[relu_input], alpha_val));
    ngraph::replace_node(m.get_match_root(), cg);
    return true;
    NGRAPH_INFO << "Boudned relu returns true!";
  };

  auto m = std::make_shared<pattern::Matcher>(min, callback, "BoundedRelu");
  this->add_matcher(m);
}
