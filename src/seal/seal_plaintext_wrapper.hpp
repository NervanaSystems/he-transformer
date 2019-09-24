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

#include <memory>

#include "seal/seal.h"

namespace ngraph {
namespace he {
/// \brief Wrapper around Seal::Plaintext with complex packing information
class SealPlaintextWrapper {
 public:
  /// \brief Constructs a SealPlaintextWrapper from a given plaintet and complex
  /// packing argument \param[in] plain Plaintext value \param[in] \param[in]
  /// complex_packing Whether or not the plaintext uses complex packing
  SealPlaintextWrapper(const seal::Plaintext& plain,
                       bool complex_packing = false)
      : m_plaintext(plain), m_complex_packing(complex_packing) {}

  /// \brief Constructs an empty plaintext wrapper using complex packing
  /// argument
  /// \brief param[in]
  SealPlaintextWrapper(bool complex_packing = false)
      : m_complex_packing(complex_packing) {}

  /// \brief Returns whether or not underlying plaintext uses complex packing
  bool complex_packing() const { return m_complex_packing; }

  /// \brief Returns whether or not underlying plaintext uses complex packing
  bool& complex_packing() { return m_complex_packing; }

  /// \brief Returns reference to underlying plaintext
  seal::Plaintext& plaintext() { return m_plaintext; }

  /// \brief Returns reference to underlying plaintext
  const seal::Plaintext& plaintext() const { return m_plaintext; }

  /// \brief Returns reference to scale of underlying plaintext
  double& scale() { return m_plaintext.scale(); }

  /// \brief Returns scale of underlying plaintext
  double scale() const { return m_plaintext.scale(); }

 private:
  seal::Plaintext m_plaintext;
  bool m_complex_packing;
};
}  // namespace he
}  // namespace ngraph
