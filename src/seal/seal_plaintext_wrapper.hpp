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

#include "he_plaintext.hpp"
#include "seal/seal.h"

namespace ngraph {
namespace runtime {
namespace he {
namespace he_seal {
struct SealPlaintextWrapper : public HEPlaintext {
 public:
  SealPlaintextWrapper(){};
  SealPlaintextWrapper(const seal::Plaintext& plain)
      : m_plaintext(plain), m_single_value(false) {}

  inline seal::Plaintext& get_hetext() { return m_plaintext; }
  inline seal::Plaintext& get_plaintext() { return m_plaintext; }
  inline const seal::Plaintext& get_plaintext() const { return m_plaintext; }

  void set_value(float f) {
    m_value = f;
    m_single_value = true;
  }

  bool is_single_value() override { return m_single_value; }

  inline float get_value() override { return m_value; }

 private:
  seal::Plaintext m_plaintext;
  float m_value;
  bool m_single_value;
};
}  // namespace he_seal
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
