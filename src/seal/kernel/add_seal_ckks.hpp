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

#include "ngraph/type/element_type.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace he {
void scalar_add_ckks(std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg0,
                     std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg1,
                     std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
                     const element::Type& element_type,
                     const ngraph::he::HESealCKKSBackend* he_seal_ckks_backend,
                     const seal::MemoryPoolHandle& pool);

void scalar_add_ckks(std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg0,
                     const HEPlaintext& arg1,
                     std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
                     const element::Type& element_type,
                     const ngraph::he::HESealCKKSBackend* he_seal_ckks_backend,
                     const seal::MemoryPoolHandle& pool);

void scalar_add_ckks(const HEPlaintext& arg0,
                     std::shared_ptr<ngraph::he::SealCiphertextWrapper>& arg1,
                     std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
                     const element::Type& element_type,
                     const ngraph::he::HESealCKKSBackend* he_seal_ckks_backend,
                     const seal::MemoryPoolHandle& pool);
}  // namespace he
}  // namespace ngraph
