//*****************************************************************************
// Copyright 2018 Intel Corporation
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

#pragma once

#include <iomanip>
#include <memory>
#include <utility>

#include "ngraph/type/element_type.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace runtime {
namespace he {
namespace he_seal {
namespace ckks {
template <typename S, typename T>
void match_scale(S* arg0, T* arg1,
                 const HESealCKKSBackend* he_seal_ckks_backend) {
  auto scale0 = arg0->get_hetext().scale();
  auto scale1 = arg1->get_hetext().scale();

  NGRAPH_ASSERT(scale0 >= 0.97 * scale1 && scale0 <= 1.02 * scale1)
      << "Scale " << std::setw(10) << scale0 << " does not match scale "
      << scale1 << " in scalar add, ratio is " << scale0 / scale1;
  arg0->get_hetext().scale() = arg1->get_hetext().scale();
}

void match_modulus_inplace(
    SealPlaintextWrapper* arg0, SealPlaintextWrapper* arg1,
    const HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

void match_modulus_inplace(
    SealPlaintextWrapper* arg0, SealCiphertextWrapper* arg1,
    const HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

void match_modulus_inplace(
    SealCiphertextWrapper* arg0, SealPlaintextWrapper* arg1,
    const HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

void match_modulus_inplace(
    SealCiphertextWrapper* arg0, SealCiphertextWrapper* arg1,
    const HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

}  // namespace ckks
}  // namespace he_seal
}  // namespace he
}  // namespace runtime
}  // namespace ngraph