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

#include <vector>

#include "he_plaintext.hpp"
#include "seal/seal.h"

namespace ngraph {
namespace runtime {
namespace he {
namespace he_seal {
struct SealPlaintextWrapper : public HEPlaintext {
 public:
  SealPlaintextWrapper(
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool())
      : m_is_encoded(false) {
    NGRAPH_INFO << "Creating plaitnext wrapper from empty arg. Is encoded? "
                << (is_encoded());
  };
  SealPlaintextWrapper(const seal::Plaintext& plain, bool encoded)
      : m_plaintext(plain), m_is_encoded(encoded) {
    NGRAPH_INFO << "Creating plaitnext wrapper from bool. Is encoded? "
                << (is_encoded());
  }

  SealPlaintextWrapper(const seal::Plaintext& plain,
                       const std::vector<float>& values)
      : HEPlaintext(values), m_plaintext(plain), m_is_encoded(false) {}

  inline seal::Plaintext& get_hetext() { return m_plaintext; }
  inline seal::Plaintext& get_plaintext() { return m_plaintext; }
  inline const seal::Plaintext& get_plaintext() const { return m_plaintext; }

  bool is_encoded() { return m_is_encoded; }
  void set_encoded(bool encoded) { m_is_encoded = encoded; }

 private:
  seal::Plaintext m_plaintext;
  bool m_is_encoded;
};
}  // namespace he_seal
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
