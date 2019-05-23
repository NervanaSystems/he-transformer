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
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {
    set_encoded(false);
    set_complex_packing(false);
  }
  SealPlaintextWrapper(const seal::Plaintext& plain, bool encoded)
      : m_plaintext(plain) {
    set_encoded(encoded);
    set_complex_packing(false);
  }

  SealPlaintextWrapper(const seal::Plaintext& plain,
                       const std::vector<float>& values)
      : HEPlaintext(values), m_plaintext(plain) {
    set_encoded(false);
    set_complex_packing(false);
  }

  inline seal::Plaintext& get_hetext() { return m_plaintext; }
  inline seal::Plaintext& get_plaintext() { return m_plaintext; }
  inline const seal::Plaintext& get_plaintext() const { return m_plaintext; }

 private:
  seal::Plaintext m_plaintext;
};

inline std::shared_ptr<runtime::he::he_seal::SealPlaintextWrapper>
cast_to_seal_hetext(std::shared_ptr<runtime::he::HEPlaintext>& plain) {
  auto seal_plaintext_wrapper =
      std::dynamic_pointer_cast<SealPlaintextWrapper>(plain);
  NGRAPH_ASSERT(seal_plaintext_wrapper != nullptr) << "Plaintext is not Seal";
  return seal_plaintext_wrapper;
};
}  // namespace he_seal
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
