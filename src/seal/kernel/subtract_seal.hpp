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
namespace runtime {
namespace he {
namespace he_seal {
namespace kernel {
void scalar_subtract(
    runtime::he::he_seal::SealCiphertextWrapper* arg0,
    runtime::he::he_seal::SealCiphertextWrapper* arg1,
    std::shared_ptr<runtime::he::he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const runtime::he::he_seal::HESealBackend* he_seal_backend);

void scalar_subtract(
    runtime::he::he_seal::SealCiphertextWrapper* arg0,
    runtime::he::he_seal::SealPlaintextWrapper* arg1,
    std::shared_ptr<runtime::he::he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const runtime::he::he_seal::HESealBackend* he_seal_backend);

void scalar_subtract(
    runtime::he::he_seal::SealPlaintextWrapper* arg0,
    runtime::he::he_seal::SealCiphertextWrapper* arg1,
    std::shared_ptr<runtime::he::he_seal::SealCiphertextWrapper>& out,
    const element::Type& element_type,
    const runtime::he::he_seal::HESealBackend* he_seal_backend);

void scalar_subtract(
    runtime::he::he_seal::SealPlaintextWrapper* arg0,
    runtime::he::he_seal::SealPlaintextWrapper* arg1,
    std::shared_ptr<runtime::he::he_seal::SealPlaintextWrapper>& out,
    const element::Type& element_type,
    const runtime::he::he_seal::HESealBackend* he_seal_backend);
}  // namespace kernel
}  // namespace he_seal
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
