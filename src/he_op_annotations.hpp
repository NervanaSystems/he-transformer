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
class HEOpAnnotations : public op::util::OpAnnotations {
 public:
  /// \brief Constructs an HE annotation.
  /// \param[in] from_client Whether or not operation should be provided by a
  /// client. This should only be set for Parameter nodes
  /// \param[in] encrypted Whether or not the output of the operation is
  /// encrypted
  /// \param[in] packed Whether or not the output of the operation is stored
  /// using plaintext packing
  HEOpAnnotations(bool from_client, bool encrypted, bool packed);

  bool from_client() const;
  void set_from_client(bool val);

  bool encrypted() const;
  void set_encrypted(bool val);

  bool packed() const;
  void set_packed(bool val);

  /// \brief Returns whether or not Op has HEOPAnnotations
  /// \param[in] op Operation to check for annotation
  static bool has_he_annotation(const op::Op& op);

  /// \brief Returns HEOpAnnotations from Op
  /// \param[in] op Operation to retrieve annotations from
  /// \throws ngraph_error if op doesn't have HEOpAnnotation
  static std::shared_ptr<HEOpAnnotations> he_op_annotation(const op::Op& op);

  /// \brief Returns whether or not operation node should be received from
  /// client. Defaults to false if op has no HEOpAnnotation.
  /// \param[in] op Graph operation, should be Constant or Parameter node
  static bool from_client(const op::Op& op);

  /// \brief Returns whether or not operation node should be packed using
  /// plaintext packing. Defaults to false if op has no HEOpAnnotation.
  /// \param[in] op Graph operation
  static bool plaintext_packed(const op::Op& op);

  static std::shared_ptr<HEOpAnnotations>
  server_plaintext_unpacked_annotation();

  static std::shared_ptr<HEOpAnnotations> server_plaintext_packed_annotation();

  static std::shared_ptr<HEOpAnnotations>
  server_ciphertext_unpacked_annotation();

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
