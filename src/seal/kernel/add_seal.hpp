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

#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace he {
void scalar_add_seal(
    SealCiphertextWrapper& arg0, SealCiphertextWrapper& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

void scalar_add_seal(
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg0,
    const HEPlaintext& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

void scalar_add_seal(
    const HEPlaintext& arg0, std::shared_ptr<SealCiphertextWrapper>& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

void scalar_add_seal(
    const HEPlaintext& arg0, const HEPlaintext& arg1, HEPlaintext& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

inline void add_seal(
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg0,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    size_t count,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool()) {
  NGRAPH_INFO << "Adding vecC + vecC => vecC";
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_add_seal(*arg0[i], *arg1[i], out[i], element_type, he_seal_backend);
  }
}

inline void add_seal(
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg0,
    const std::vector<std::unique_ptr<HEPlaintext>>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    size_t count,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool()) {
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_add_seal(arg0[i], *arg1[i], out[i], element_type, he_seal_backend,
                    pool);
  }
}

inline void add_seal(
    const std::vector<std::unique_ptr<HEPlaintext>>& arg0,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    size_t count,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool()) {
  add_seal(arg1, arg0, out, element_type, he_seal_backend, count, pool);
}

inline void add_seal(
    std::vector<std::unique_ptr<HEPlaintext>>& arg0,
    std::vector<std::unique_ptr<HEPlaintext>>& arg1,
    std::vector<std::unique_ptr<HEPlaintext>>& out,
    const element::Type& element_type, const HESealBackend* he_seal_backend,
    size_t count,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool()) {
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_add_seal(*arg0[i], *arg1[i], *out[i], element_type, he_seal_backend,
                    pool);
  }
}
}  // namespace he
}  // namespace ngraph
