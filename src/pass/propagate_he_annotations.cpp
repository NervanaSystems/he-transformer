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

  auto plaintext_unpacked_annotation =
      std::make_shared<HEOpAnnotations>(false, false, false);

  // First, set all ops without annotations to have plaintext unpacked
  // annotation
  for (auto node : nodes) {
    NGRAPH_HE_LOG(5) << "Node " << node->get_name();
    if (node->is_op()) {
      auto op = std::dynamic_pointer_cast<ngraph::op::Op>(node);
      NGRAPH_INFO << "Node is op";
      if (!ngraph::he::HEOpAnnotations::has_he_annotation(*op)) {
        op->set_op_annotations(plaintext_unpacked_annotation);
        NGRAPH_HE_LOG(5) << "Adding plaintext_unpacked_annotation to op "
                         << op->get_name();
      } else {
        auto he_op_annotations = std::dynamic_pointer_cast<HEOpAnnotations>(
            op->get_op_annotations());
        NGRAPH_HE_LOG(5) << "Op has annotation : " << *he_op_annotations;
      }
    }
  }

  // Update annotation with ciphertext / packed as needed
  for (auto node : nodes) {
    auto op = std::dynamic_pointer_cast<ngraph::op::Op>(node);
    if (op == nullptr) {
      continue;
    }
    auto he_op_annotations =
        std::dynamic_pointer_cast<HEOpAnnotations>(op->get_op_annotations());
    NGRAPH_CHECK(he_op_annotations != nullptr,
                 "Node doesn't have HEOpAnnotations");
    for (ngraph::Output<ngraph::Node> output : op->outputs()) {
      ngraph::Node* out_node = output.get_node();

      auto out_op = dynamic_cast<ngraph::op::Op*>(out_node);
      if (out_op == nullptr) {
        continue;
      }

      auto he_output_annotations = std::dynamic_pointer_cast<HEOpAnnotations>(
          out_op->get_op_annotations());
      NGRAPH_CHECK(he_output_annotations != nullptr,
                   "Output node doesn't have HEOpAnnotations");

      if (he_op_annotations->encrypted()) {
        NGRAPH_HE_LOG(5) << "Setting node " << node->get_name()
                         << " to encrypted";
        he_output_annotations->set_encrypted(true);
      }
      if (he_op_annotations->packed()) {
        NGRAPH_HE_LOG(5) << "Setting node " << node->get_name() << " to packed";
        he_output_annotations->set_packed(true);
      }
      out_op->set_op_annotations(he_output_annotations);
    }
  }
  return false;
}

}  // namespace he
}  // namespace ngraph