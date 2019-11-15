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

#include "ngraph/pass/graph_rewrite.hpp"

namespace ngraph::runtime::he::pass {

/// \brief performs HE-friendly fusion operations
class HEFusion : public ngraph::pass::GraphRewrite {
 public:
  HEFusion() : GraphRewrite() { construct_bounded_relu(); }

  /// \brief Fuses Min(Relu, Constant) op into BoundedRelu(Constant) op
  void construct_bounded_relu();
};
}  // namespace ngraph::runtime::he::pass
