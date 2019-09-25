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

#include <functional>
#include <memory>
#include <string>

#include "ngraph/pass/graph_rewrite.hpp"

namespace ngraph {
namespace he {
namespace pass {
/// \brief Checks whether the graph contains any ops that are not supported
class SupportedOps : public ngraph::pass::FunctionPass {
 public:
  /// \param[in] is_supported Function which returns whether or not a given Node
  /// is supported
  SupportedOps(std::function<bool(const ngraph::Node&)> is_supported)
      : m_is_supported(is_supported) {}

  /// \brief Returns true if function is supported
  /// \throws ngraph_error is function is not supported
  /// \param[in,out] function Function which to run pass on
  bool run_on_function(std::shared_ptr<ngraph::Function> function) override;

  /// \brief returns whether or not given node is supported
  /// \param[in] node Node which to check supported status
  /// \return true if node is supported, false otherwise
  bool is_supported(const ngraph::Node& node) const {
    return m_is_supported(node);
  }

 private:
  std::function<bool(const ngraph::Node&)> m_is_supported;
};
}  // namespace pass
}  // namespace he
}  // namespace ngraph
