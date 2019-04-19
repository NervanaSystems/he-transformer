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
      : m_is_encoded(false){};
  SealPlaintextWrapper(const seal::Plaintext& plain, bool is_encoded)
      : m_plaintext(plain), m_is_encoded(is_encoded) {}

  SealPlaintextWrapper(const seal::Plaintext& plain,
                       const std::vector<float>& values)
      : HEPlaintext(values), m_plaintext(plain) {}

  inline seal::Plaintext& get_hetext() { return m_plaintext; }
  inline seal::Plaintext& get_plaintext() { return m_plaintext; }
  inline const seal::Plaintext& get_plaintext() const { return m_plaintext; }

  bool is_encoded() { return m_is_encoded; }

 private:
  seal::Plaintext m_plaintext;
  bool m_is_encoded;
};
}  // namespace he_seal
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
