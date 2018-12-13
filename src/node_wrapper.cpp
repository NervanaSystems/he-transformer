//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "node_wrapper.hpp"

using namespace ngraph;
using namespace std;

runtime::he::NodeWrapper::NodeWrapper(const shared_ptr<const Node>& node)
    : m_node{node} {
// This expands the op list in op_tbl.hpp into a list of enumerations that look
// like this:
// {"Abs", runtime::he::OP_TYPEID::Abs},
// {"Acos", runtime::he::OP_TYPEID::Acos},
// ...
#define NGRAPH_OP(a, b) {#a, runtime::he::OP_TYPEID::a},
  static unordered_map<string, runtime::he::OP_TYPEID> typeid_map{
#include "ngraph/op/op_tbl.hpp"
  };
#undef NGRAPH_OP

  // TODO: figure out why m_node->description() is empty.
  // Probably related to using ngraph-tensorflow-bridge v0.8.0, and ng v0.11.0
  auto name = m_node->get_name();
  auto pos = name.rfind('_');
  if (pos != std::string::npos) {
    name.erase(pos);
  }
  NGRAPH_INFO << "New name " << name;
  auto it = typeid_map.find(name);
  if (it != typeid_map.end()) {
    m_typeid = it->second;
  } else {
    NGRAPH_INFO << "unsupported op get_name()" << m_node->get_name();
    NGRAPH_INFO << "m_node->get_friendly_name() "
                << m_node->get_friendly_name();
    NGRAPH_INFO << "m_node->get_friendly_name() "
                << m_node->get_friendly_name();
    throw unsupported_op("Unsupported op '" + m_node->description() + "'");
  }
}