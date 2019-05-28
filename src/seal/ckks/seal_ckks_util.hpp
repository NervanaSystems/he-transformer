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

#include <cmath>
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
template <typename S>
size_t get_chain_index(S* hetext,
                       const HESealCKKSBackend* he_seal_ckks_backend) {
  size_t chain_ind = he_seal_ckks_backend->get_context()
                         ->context_data(hetext->get_hetext().parms_id())
                         ->chain_index();
  return chain_ind;
}

template <typename S, typename T>
bool within_rescale_tolerance(const S* arg0, const T* arg1,
                              double factor = 1.02) {
  const auto scale0 = arg0->get_hetext().scale();
  const auto scale1 = arg1->get_hetext().scale();

  bool within_tolerance =
      (scale0 / scale1 <= factor && scale1 / scale0 <= factor);
  return within_tolerance;
}

template <typename S, typename T>
void match_scale(S* arg0, T* arg1,
                 const HESealCKKSBackend* he_seal_ckks_backend) {
  auto scale0 = arg0->get_hetext().scale();
  auto scale1 = arg1->get_hetext().scale();
  NGRAPH_ASSERT(within_rescale_tolerance(arg0, arg1))
      << "Scale " << std::setw(10) << scale0 << " does not match scale "
      << scale1 << " in scalar add, ratio is " << scale0 / scale1;
  arg0->get_hetext().scale() = arg1->get_hetext().scale();
}

void match_modulus_and_scale_inplace(
    SealCiphertextWrapper* arg0, SealCiphertextWrapper* arg1,
    const HESealCKKSBackend* he_seal_ckks_backend,
    seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());

void multiply_by_double_inplace(
    seal::Ciphertext& encrypted, double value,
    const HESealCKKSBackend* he_seal_ckks_backend,
    seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());

inline void multiply_by_double(
    const seal::Ciphertext& encrypted, double value,
    seal::Ciphertext& destination,
    const HESealCKKSBackend* he_seal_ckks_backend,
    seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {
  destination = encrypted;
  ngraph::runtime::he::he_seal::ckks::multiply_by_double_inplace(
      destination, value, he_seal_ckks_backend, std::move(pool));
}
}  // namespace ckks
}  // namespace he_seal
}  // namespace he
}  // namespace runtime
}  // namespace ngraph