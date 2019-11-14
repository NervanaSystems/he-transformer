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

#include "he_op_annotations.hpp"

#include <functional>
#include <iostream>
#include <memory>

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/op_annotations.hpp"

namespace ngraph::runtime::he {

HEOpAnnotations::HEOpAnnotations(bool from_client, bool encrypted, bool packed)
    : m_from_client(from_client), m_encrypted(encrypted), m_packed(packed) {}

HEOpAnnotations::HEOpAnnotations(const HEOpAnnotations& op_annotation) =
    default;

bool HEOpAnnotations::operator==(const HEOpAnnotations& other) const {
  return (m_from_client == other.m_from_client) &&
         (m_encrypted == other.m_encrypted) && (m_packed == other.m_packed);
}

bool HEOpAnnotations::from_client() const { return m_from_client; }
void HEOpAnnotations::set_from_client(bool val) { m_from_client = val; }

bool HEOpAnnotations::encrypted() const { return m_encrypted; }
void HEOpAnnotations::set_encrypted(bool val) { m_encrypted = val; }

bool HEOpAnnotations::packed() const { return m_packed; }
void HEOpAnnotations::set_packed(bool val) { m_packed = val; }

bool HEOpAnnotations::has_he_annotation(const op::Op& op) {
  auto annotation = op.get_op_annotations();
  return std::dynamic_pointer_cast<HEOpAnnotations>(annotation) != nullptr;
}

std::shared_ptr<HEOpAnnotations> HEOpAnnotations::he_op_annotation(
    const op::Op& op) {
  NGRAPH_CHECK(HEOpAnnotations::has_he_annotation(op), "op ", op.get_name(),
               " has no HEOpAnnotation");
  return std::static_pointer_cast<HEOpAnnotations>(op.get_op_annotations());
}

bool HEOpAnnotations::from_client(const op::Op& op) {
  auto annotation = op.get_op_annotations();
  if (auto he_annotation =
          std::dynamic_pointer_cast<HEOpAnnotations>(annotation)) {
    return he_annotation->from_client();
  }
  return false;
}

bool HEOpAnnotations::plaintext_packed(const op::Op& op) {
  auto annotation = op.get_op_annotations();
  if (auto he_annotation =
          std::dynamic_pointer_cast<HEOpAnnotations>(annotation)) {
    return he_annotation->packed();
  }
  return false;
}

std::shared_ptr<HEOpAnnotations>
HEOpAnnotations::server_plaintext_unpacked_annotation() {
  return std::make_shared<HEOpAnnotations>(false, false, false);
}

std::shared_ptr<HEOpAnnotations>
HEOpAnnotations::server_ciphertext_unpacked_annotation() {
  return std::make_shared<HEOpAnnotations>(false, true, false);
}

std::ostream& operator<<(std::ostream& os, const HEOpAnnotations& annotation) {
  os << "HEOpAnnotation{";
  os << "from_client=" << (annotation.from_client() ? "True" : "False") << ", ";
  os << "encrypted=" << (annotation.encrypted() ? "True" : "False") << ", ";
  os << "packed=" << (annotation.packed() ? "True" : "False") << "}";
  return os;
}

}  // namespace ngraph::runtime::he
