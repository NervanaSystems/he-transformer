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

#include "ngraph/op/util/op_annotations.hpp"

namespace ngraph {
namespace he {
/// \brief Annotations added to graph ops by CPU backend passes
class HEOpAnnotations : public ngraph::op::util::OpAnnotations {
 public:
  HEOpAnnotations(bool encrypted) : m_encrypted(encrypted) {}
  inline bool is_encrypted() { return m_encrypted; }
  inline void set_encrypted(bool val) { m_encrypted = val; }

 private:
  bool m_encrypted = false;
};

}  // namespace he
}  // namespace ngraph