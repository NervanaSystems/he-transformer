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

#include "he_ciphertext.hpp"
#include "he_plaintext.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace he {
void scalar_multiply(
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg0,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const ngraph::he::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

void scalar_multiply(
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg0,
    std::shared_ptr<ngraph::he::HEPlaintext>& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const ngraph::he::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

void scalar_multiply(
    std::shared_ptr<ngraph::he::HEPlaintext>& arg0,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const ngraph::he::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

void scalar_multiply(
    std::shared_ptr<ngraph::he::HEPlaintext>& arg0,
    std::shared_ptr<ngraph::he::HEPlaintext>& arg1,
    std::shared_ptr<ngraph::he::HEPlaintext>& out,
    const element::Type& element_type,
    const ngraph::he::HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

template <typename S, typename T, typename V>
void scalar_multiply(
    std::vector<std::shared_ptr<S>>& arg0,
    std::vector<std::shared_ptr<T>>& arg1, std::vector<std::shared_ptr<V>>& out,
    const element::Type& element_type, const ngraph::he::HEBackend* he_backend,
    size_t count,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool()) {
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_multiply(arg0[i], arg1[i], out[i], element_type, he_backend, pool);
  }
}
}  // namespace he
}  // namespace ngraph
