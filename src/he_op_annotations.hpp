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

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/op_annotations.hpp"

namespace ngraph {
namespace he {
/// \brief Annotations added to graph ops by HE backend passes
class HEOpAnnotations : public ngraph::op::util::OpAnnotations {
 public:
  HEOpAnnotations(bool from_client, bool encrypted, bool plaintext_packing);

  bool from_client();
  void set_from_client(bool val);

  bool encrypted();
  void set_encrypted(bool val);

  bool plaintext_packing();
  void set_plaintext_packing(bool val);

 private:
  bool m_from_client = false;
  bool m_encrypted = false;
  bool m_plaintext_packing = false;
};

/// \brief Returns whether or not Op has HEOPAnnotations
/// \param[in] op Operation to check for annotation
bool has_he_annotation(const ngraph::op::Op& op);

}  // namespace he
}  // namespace ngraph