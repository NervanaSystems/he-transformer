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
#include "seal/kernel/negate_seal.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace he {

/// \brief Adds two ciphertexts
/// \para[in,out] arg0 Ciphertext argument to add
/// \param[in,out] arg1 Ciphertext rgument to add
/// \param[out] out Stores the sum
/// \param[in] element_type datatype of the addition
/// \param[in] he_seal_backend Backend with which to perform addition
/// \param[in] pool Memory pool with which to perform addition
void scalar_add_seal(
    SealCiphertextWrapper& arg0, SealCiphertextWrapper& arg1,
    std::shared_ptr<SealCiphertextWrapper>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

void scalar_add_seal(
    SealCiphertextWrapper& arg0, const HEPlaintext& arg1,
    std::shared_ptr<SealCiphertextWrapper>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

inline void scalar_add_seal(
    const HEPlaintext& arg0, SealCiphertextWrapper& arg1,
    std::shared_ptr<SealCiphertextWrapper>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool()) {
  ngraph::he::scalar_add_seal(arg1, arg0, out, element_type, he_seal_backend);
}

void scalar_add_seal(const HEPlaintext& arg0, const HEPlaintext& arg1,
                     HEPlaintext& out, const element::Type& element_type,
                     HESealBackend& he_seal_backend);

inline void add_seal(
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg0,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    size_t count,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool()) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_add_seal(*arg0[i], *arg1[i], out[i], element_type, he_seal_backend);
  }
}

inline void add_seal(
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg0,
    const std::vector<HEPlaintext>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    size_t count,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool()) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_add_seal(*arg0[i], arg1[i], out[i], element_type, he_seal_backend,
                    pool);
  }
}

inline void add_seal(
    const std::vector<HEPlaintext>& arg0,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    size_t count,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool()) {
  add_seal(arg1, arg0, out, element_type, he_seal_backend, count, pool);
}

inline void add_seal(std::vector<HEPlaintext>& arg0,
                     std::vector<HEPlaintext>& arg1,
                     std::vector<HEPlaintext>& out,
                     const element::Type& element_type,
                     HESealBackend& he_seal_backend, size_t count) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_add_seal(arg0[i], arg1[i], out[i], element_type, he_seal_backend);
  }
}
}  // namespace he
}  // namespace ngraph
