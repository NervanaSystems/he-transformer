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

#include "pass/supported_ops.hpp"

#include <list>

#include "ngraph/check.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/util.hpp"

namespace ngraph::he {

bool pass::SupportedOps::run_on_function(
    std::shared_ptr<ngraph::Function> function) {
  std::list<std::shared_ptr<ngraph::Node>> ops = function->get_ordered_ops();

  for (const auto& op : ops) {
    NGRAPH_CHECK(is_supported(*op), "Unsupported op ", op->description(),
                 " with type ", op->get_element_type());
  }
  return true;
}

}  // namespace ngraph::he
