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

#include "he_ciphertext.hpp"
#include "seal/seal.h"

namespace ngraph {
namespace he {
struct SealCiphertextWrapper : public HECiphertext {
  SealCiphertextWrapper() {
    set_complex_packing(false);
    set_zero(false);
  }

  SealCiphertextWrapper(const seal::Ciphertext& cipher) : m_ciphertext(cipher) {
    set_complex_packing(false);
    set_zero(false);
  }

  seal::Ciphertext& get_hetext() { return m_ciphertext; }
  const seal::Ciphertext& get_hetext() const { return m_ciphertext; }

  void save(std::ostream& stream) const override { m_ciphertext.save(stream); }

  size_t size() const override { return m_ciphertext.size(); }

  seal::Ciphertext m_ciphertext;
};

inline std::shared_ptr<ngraph::he::SealCiphertextWrapper> cast_to_seal_hetext(
    std::shared_ptr<ngraph::he::HECiphertext>& cipher) {
  auto seal_ciphertext_wrapper =
      std::dynamic_pointer_cast<SealCiphertextWrapper>(cipher);
  NGRAPH_ASSERT(seal_ciphertext_wrapper != nullptr) << "Ciphertext is not Seal";
  return seal_ciphertext_wrapper;
}

inline const std::shared_ptr<ngraph::he::SealCiphertextWrapper>
cast_to_seal_hetext(const std::shared_ptr<ngraph::he::HECiphertext>& cipher) {
  auto seal_ciphertext_wrapper =
      std::dynamic_pointer_cast<SealCiphertextWrapper>(cipher);
  NGRAPH_ASSERT(seal_ciphertext_wrapper != nullptr) << "Ciphertext is not Seal";
  return seal_ciphertext_wrapper;
};
}  // namespace he
}  // namespace ngraph
