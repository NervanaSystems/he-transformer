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
struct SealCiphertextWrapper {
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

  void save(std::ostream& stream) const { m_ciphertext.save(stream); }

  size_t size() const { return m_ciphertext.size(); }

  bool is_zero() const { return m_is_zero; }
  void set_zero(bool toggle) { m_is_zero = toggle; }

  bool complex_packing() const { return m_complex_packing; }
  void set_complex_packing(bool toggle) { m_complex_packing = toggle; }

 private:
  bool m_complex_packing;
  bool m_is_zero;
  seal::Ciphertext m_ciphertext;
};

}  // namespace he
}  // namespace ngraph
