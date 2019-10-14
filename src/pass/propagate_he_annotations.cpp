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

#include <exception>
#include <sstream>
#include <unordered_set>

#include "he_op_annotations.hpp"
#include "logging/ngraph_he_log.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/util.hpp"
#include "pass/propagate_he_annotations.hpp"

namespace ngraph {
namespace he {

bool pass::PropagateHEAnnotations::run_on_function(
    std::shared_ptr<ngraph::Function> function) {
  std::list<std::shared_ptr<ngraph::Node>> nodes = function->get_ordered_ops();

  NGRAPH_HE_LOG(3) << "Running Propagate HE Annotations pass";

  // First, set all ops without annotations to have plaintext unpacked
  // annotation
  for (auto node : nodes) {
    if (node->is_op()) {
      auto op = std::dynamic_pointer_cast<ngraph::op::Op>(node);
      if (!ngraph::he::HEOpAnnotations::has_he_annotation(*op)) {
        op->set_op_annotations(
            HEOpAnnotations::server_plaintext_unpacked_annotation());
        NGRAPH_HE_LOG(5) << "Adding server plaintext_unpacked_annotation to op "
                         << op->get_name();
      } else {
        auto he_op_annotations = std::dynamic_pointer_cast<HEOpAnnotations>(
            op->get_op_annotations());
        NGRAPH_HE_LOG(5) << "Op has annotation: " << *he_op_annotations;
      }
    } else {
      NGRAPH_HE_LOG(5) << "Node " << node->get_name() << " is not an op";
    }
  }

  NGRAPH_HE_LOG(5) << "Updating annotation with ciphertext / packed as "
                      "needed";
  // Node has encrypted output if any of its inputs is encrypted
  // Node has packed output if any of its inputs is packed
  for (auto node : nodes) {
    auto op = std::dynamic_pointer_cast<ngraph::op::Op>(node);
    if (op == nullptr) {
      NGRAPH_HE_LOG(5) << "Node " << node->get_name() << " is not op";
      continue;
    }
    NGRAPH_HE_LOG(5) << "Op " << op->get_name();
    auto he_op_annotations = HEOpAnnotations::he_op_annotation(*op);
    NGRAPH_HE_LOG(5) << "Annotation " << *he_op_annotations;

    for (const auto& output : node->outputs()) {
      for (const auto& target_input : output.get_target_inputs()) {
        auto target_node = target_input.get_node();
        auto target_op = dynamic_cast<ngraph::op::Op*>(target_node);
        NGRAPH_CHECK(target_op != nullptr, "Target is not an op");

        auto he_target_annotations = std::dynamic_pointer_cast<HEOpAnnotations>(
            target_op->get_op_annotations());
        NGRAPH_CHECK(he_target_annotations != nullptr, "Target node ",
                     target_op->get_name(), " doesn't have HEOpAnnotations");
        NGRAPH_HE_LOG(5) << "Target node " << target_op->get_name()
                         << " has HE annotation " << *he_target_annotations;

        if (he_op_annotations->encrypted()) {
          NGRAPH_HE_LOG(5) << "Setting node " << target_node->get_name()
                           << " to encrypted";
          he_target_annotations->set_encrypted(true);
        }
        if (he_op_annotations->packed()) {
          NGRAPH_HE_LOG(5) << "Setting node " << target_node->get_name()
                           << " to packed";
          he_target_annotations->set_packed(true);
        }
        target_op->set_op_annotations(he_target_annotations);
      }
    }
  }

  // For debugging, print out node info
  NGRAPH_HE_LOG(5) << "Final node annotations";
  for (auto node : nodes) {
    if (node->is_op()) {
      auto op = std::dynamic_pointer_cast<ngraph::op::Op>(node);
      if (ngraph::he::HEOpAnnotations::has_he_annotation(*op)) {
        auto he_op_annotations = std::dynamic_pointer_cast<HEOpAnnotations>(
            op->get_op_annotations());
        NGRAPH_HE_LOG(5) << "Op " << op->get_name() << " (" << op->get_shape()
                         << ") has annotation: " << *he_op_annotations;
      }
    } else {
      NGRAPH_HE_LOG(5) << "Node " << node->get_name() << " is not an op";
    }
  }
  return false;
}

}  // namespace he
}  // namespace ngraph