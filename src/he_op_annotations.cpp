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

#include <functional>
#include <memory>

#include "he_op_annotations.hpp"

namespace ngraph {
namespace he {
HEOpAnnotations::HEOpAnnotations(bool from_client, bool encrypted,
                                 bool plaintext_packing)
    : m_from_client(from_client),
      m_encrypted(encrypted),
      m_plaintext_packing(plaintext_packing) {}

bool HEOpAnnotations::from_client() { return m_from_client; }

void HEOpAnnotations::set_from_client(bool val) { m_from_client = val; }

bool HEOpAnnotations::encrypted() { return m_encrypted; }
void HEOpAnnotations::set_encrypted(bool val) { m_encrypted = val; }

bool HEOpAnnotations::plaintext_packing() { return m_plaintext_packing; }
void HEOpAnnotations::set_plaintext_packing(bool val) {
  m_plaintext_packing = val;
}

/// \brief Returns whether or not Op has HEOPAnnotations
/// \param[in] op Operation to check for annotation
bool has_he_annotation(const ngraph::op::Op& op) {
  auto annotation = op.get_op_annotations();
  return std::dynamic_pointer_cast<HEOpAnnotations>(annotation) != nullptr;
}

}  // namespace he
}  // namespace ngraph
