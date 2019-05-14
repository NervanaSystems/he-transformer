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

#include "he_ciphertext.hpp"
#include "seal/seal.h"

namespace ngraph {
namespace runtime {
namespace he {
namespace he_seal {
struct SealCiphertextWrapper : public HECiphertext {
  SealCiphertextWrapper() : m_complex_packed(false){};
  SealCiphertextWrapper(const seal::Ciphertext& cipher)
      : m_ciphertext(cipher), m_complex_packed(false) {}

  seal::Ciphertext& get_hetext() { return m_ciphertext; }

  void save(std::ostream& stream) const override { m_ciphertext.save(stream); }

  bool complex_packing() const override { return m_complex_packed; }

  void set_complex_packing(bool toggle) override { m_complex_packed = toggle; }

  seal::Ciphertext m_ciphertext;
  bool m_complex_packed;
};
}  // namespace he_seal
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
