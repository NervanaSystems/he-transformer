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
#include <iostream>
#include <memory>

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/op_annotations.hpp"

namespace ngraph::he {
/// \brief Annotations added to graph ops by HE backend passes
class HEOpAnnotations : public ngraph::op::util::OpAnnotations {
 public:
  /// \brief Constructs an HE annotation.
  /// \param[in] from_client Whether or not operation should be provided by a
  /// client. This should only be set for Parameter nodes
  /// \param[in] encrypted Whether or not the output of the operation is
  /// encrypted
  /// \param[in] packed Whether or not the output of the operation is stored
  /// using plaintext packing
  HEOpAnnotations(bool from_client, bool encrypted, bool packed)
      : m_from_client(from_client), m_encrypted(encrypted), m_packed(packed) {}

  bool from_client() const { return m_from_client; }
  void set_from_client(bool val) { m_from_client = val; }

  bool encrypted() const { return m_encrypted; }
  void set_encrypted(bool val) { m_encrypted = val; }

  bool packed() const { return m_packed; }
  void set_packed(bool val) { m_packed = val; }

  /// \brief Returns whether or not Op has HEOPAnnotations
  /// \param[in] op Operation to check for annotation
  static bool has_he_annotation(const ngraph::op::Op& op) {
    auto annotation = op.get_op_annotations();
    return std::dynamic_pointer_cast<HEOpAnnotations>(annotation) != nullptr;
  }

  /// \brief Returns HEOpAnnotations from Op
  /// \param[in] op Operation to retrieve annotations from
  /// \throws ngraph_error if op doesn't have HEOpAnnotation
  static std::shared_ptr<HEOpAnnotations> he_op_annotation(
      const ngraph::op::Op& op) {
    NGRAPH_CHECK(HEOpAnnotations::has_he_annotation(op), "op ", op.get_name(),
                 " has no HEOpAnnotation");
    return std::static_pointer_cast<HEOpAnnotations>(op.get_op_annotations());
  }

  static std::shared_ptr<HEOpAnnotations>
  server_plaintext_unpacked_annotation() {
    return std::make_shared<HEOpAnnotations>(false, false, false);
  }

  static std::shared_ptr<HEOpAnnotations> server_plaintext_packed_annotation() {
    return std::make_shared<HEOpAnnotations>(false, false, true);
  }

  static std::shared_ptr<HEOpAnnotations>
  server_ciphertext_unpacked_annotation() {
    return std::make_shared<HEOpAnnotations>(false, true, false);
  }

  static std::shared_ptr<HEOpAnnotations>
  server_ciphertext_packed_annotation() {
    return std::make_shared<HEOpAnnotations>(false, true, true);
  }

  static std::shared_ptr<HEOpAnnotations>
  client_plaintext_unpacked_annotation() {
    return std::make_shared<HEOpAnnotations>(true, false, false);
  }

  static std::shared_ptr<HEOpAnnotations> client_plaintext_packed_annotation() {
    return std::make_shared<HEOpAnnotations>(true, false, true);
  }

  static std::shared_ptr<HEOpAnnotations>
  client_ciphertext_unpacked_annotation() {
    return std::make_shared<HEOpAnnotations>(true, true, false);
  }

  static std::shared_ptr<HEOpAnnotations>
  client_ciphertext_packed_annotation() {
    return std::make_shared<HEOpAnnotations>(true, true, true);
  }

 private:
  bool m_from_client = false;
  bool m_encrypted = false;
  bool m_packed = false;
};

inline std::ostream& operator<<(std::ostream& os,
                                const HEOpAnnotations& annotation) {
  os << "HEOpAnnotation{";
  os << "from_client=" << (annotation.from_client() ? "True" : "False") << ", ";
  os << "encrypted=" << (annotation.encrypted() ? "True" : "False") << ", ";
  os << "packed=" << (annotation.packed() ? "True" : "False") << "}";
  return os;
}

}  // namespace ngraph::he
