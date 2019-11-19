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

#include "seal/seal_plaintext_wrapper.hpp"

#include <utility>

#include "seal/seal.h"

namespace ngraph::runtime::he {

SealPlaintextWrapper::SealPlaintextWrapper(seal::Plaintext plain,
                                           bool complex_packing)
    : m_plaintext(std::move(plain)), m_complex_packing(complex_packing) {}

SealPlaintextWrapper::SealPlaintextWrapper(bool complex_packing)
    : m_complex_packing(complex_packing) {}

bool SealPlaintextWrapper::complex_packing() const { return m_complex_packing; }

bool& SealPlaintextWrapper::complex_packing() { return m_complex_packing; }

const seal::Plaintext& SealPlaintextWrapper::plaintext() const {
  return m_plaintext;
}

seal::Plaintext& SealPlaintextWrapper::plaintext() { return m_plaintext; }

}  // namespace ngraph::runtime::he
