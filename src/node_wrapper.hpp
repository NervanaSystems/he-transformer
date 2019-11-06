
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

#pragma once

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph::he {
enum class OP_TYPEID;
class NodeWrapper;
}  // namespace ngraph::he
// This expands the op list in op_tbl.hpp into a list of enumerations that look
// like this: Abs, Acos,
// ...
#define NGRAPH_OP(a, b) a,
enum class ngraph::he::OP_TYPEID {
#include "ngraph/op/op_tbl.hpp"
  NGRAPH_OP(BoundedRelu, ngraph::op)
};
#undef NGRAPH_OP

/// \brief This class allows adding an enum typeid to each Node. This makes
/// dealing with collections of Nodes a little easier and faster as we can use
/// switch() instead of if/else statements
namespace ngraph::he {
class NodeWrapper {
 public:
  explicit NodeWrapper(std::shared_ptr<const ngraph::Node> node);

  std::shared_ptr<const Node> get_node() const { return m_node; }
  ngraph::he::OP_TYPEID get_typeid() const { return m_typeid; }

  std::shared_ptr<const ngraph::op::Op> get_op() const;

 private:
  std::shared_ptr<const ngraph::Node> m_node;
  OP_TYPEID m_typeid;
};
}  // namespace ngraph::he