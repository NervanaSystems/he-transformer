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
//*****************************************************************************

#pragma once

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
namespace kernel {
void scalar_multiply_ckks(
    const std::shared_ptr<const he_seal::SealCiphertextWrapper>& arg0,
    const std::shared_ptr<const he_seal::SealCiphertextWrapper>& arg1,
    std::shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealCKKSBackend* he_seal_ckks_backend);

void scalar_multiply_ckks(
    const std::shared_ptr<const he_seal::SealCiphertextWrapper>& arg0,
    const std::shared_ptr<const he_seal::SealPlaintextWrapper>& arg1,
    std::shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealCKKSBackend* he_seal_ckks_backend);

void scalar_multiply_ckks(
    const std::shared_ptr<const he_seal::SealPlaintextWrapper>& arg0,
    const std::shared_ptr<const he_seal::SealCiphertextWrapper>& arg1,
    std::shared_ptr<he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const he_seal::HESealCKKSBackend* he_seal_ckks_backend);
}  // namespace kernel
}  // namespace ckks
}  // namespace he_seal
}  // namespace he
}  // namespace runtime
}  // namespace ngraph