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
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace he {
void scalar_multiply_seal(
    SealCiphertextWrapper& arg0, SealCiphertextWrapper& arg1,
    std::shared_ptr<SealCiphertextWrapper>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

void scalar_multiply_seal(
    SealCiphertextWrapper& arg0, const HEPlaintext& arg1,
    std::shared_ptr<SealCiphertextWrapper>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

inline void scalar_multiply_seal(
    const HEPlaintext& arg0, SealCiphertextWrapper& arg1,
    std::shared_ptr<SealCiphertextWrapper>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool()) {
  scalar_multiply_seal(arg1, arg0, out, element_type, he_seal_backend, pool);
}

void scalar_multiply_seal(
    const HEPlaintext& arg0, const HEPlaintext& arg1, HEPlaintext& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

inline void multiply_seal(
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg0,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    size_t count,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool()) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type", element_type);
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_multiply_seal(*arg0[i], *arg1[i], out[i], element_type,
                         he_seal_backend);
  }
}

inline void multiply_seal(
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg0,
    const std::vector<HEPlaintext>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    size_t count,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool()) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type", element_type);
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_multiply_seal(*arg0[i], arg1[i], out[i], element_type,
                         he_seal_backend, pool);
  }
}

inline void multiply_seal(
    const std::vector<HEPlaintext>& arg0,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    size_t count,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool()) {
  multiply_seal(arg1, arg0, out, element_type, he_seal_backend, count, pool);
}

inline void multiply_seal(const std::vector<HEPlaintext>& arg0,
                          const std::vector<HEPlaintext>& arg1,
                          std::vector<HEPlaintext>& out,
                          const element::Type& element_type,
                          HESealBackend& he_seal_backend, size_t count) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type", element_type);
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_multiply_seal(arg0[i], arg1[i], out[i], element_type,
                         he_seal_backend);
  }
}
}  // namespace he
}  // namespace ngraph
