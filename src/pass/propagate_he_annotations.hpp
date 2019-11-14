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

namespace ngraph::runtime::he::pass {
/// \brief Propagates HE op annotations from parameters / constants to entire
/// function
class PropagateHEAnnotations : public ngraph::pass::FunctionPass {
 public:
  /// \brief Runs pass on function
  /// \param[in,out] function Function which to run pass on
  /// \returns whether or not the function has been modified
  bool run_on_function(std::shared_ptr<Function> function) override;
};
}  // namespace ngraph::runtime::he::pass
