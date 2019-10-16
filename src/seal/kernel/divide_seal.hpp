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
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace he {
void scalar_divide_seal(SealCiphertextWrapper& arg0,
                        SealCiphertextWrapper& arg1,
                        std::shared_ptr<SealCiphertextWrapper>& out,
                        const element::Type& element_type,
                        HESealBackend& he_seal_backend);

void scalar_divide_seal(SealCiphertextWrapper& arg0, const HEPlaintext& arg1,
                        std::shared_ptr<SealCiphertextWrapper>& out,
                        const element::Type& element_type,
                        HESealBackend& he_seal_backend);

void scalar_divide_seal(const HEPlaintext& arg0, SealCiphertextWrapper& arg1,
                        std::shared_ptr<SealCiphertextWrapper>& out,
                        const element::Type& element_type,
                        HESealBackend& he_seal_backend);

void scalar_divide_seal(const HEPlaintext& arg0, const HEPlaintext& arg1,
                        HEPlaintext& out);

inline void divide_seal(
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
    scalar_divide_seal(*arg0[i], *arg1[i], out[i], element_type,
                       he_seal_backend);
  }
}

inline void divide_seal(
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg0,
    const std::vector<HEPlaintext>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    size_t count) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_divide_seal(*arg0[i], arg1[i], out[i], element_type,
                       he_seal_backend);
  }
}

inline void divide_seal(
    const std::vector<HEPlaintext>& arg0,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    size_t count) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_divide_seal(arg0[i], *arg1[i], out[i], element_type,
                       he_seal_backend);
  }
}

inline void divide_seal(std::vector<HEPlaintext>& arg0,
                        std::vector<HEPlaintext>& arg1,
                        std::vector<HEPlaintext>& out, size_t count) {
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_divide_seal(arg0[i], arg1[i], out[i]);
  }
}
}  // namespace he
}  // namespace ngraph
