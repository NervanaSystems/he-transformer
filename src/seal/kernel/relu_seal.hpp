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

#include <vector>

#include "he_type.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_util.hpp"

namespace ngraph::he {
void scalar_relu_seal(const HEPlaintext& arg, HEPlaintext& out);

void scalar_relu_seal(const HEType& arg, HEType& out,
                      const seal::parms_id_type& parms_id, double scale,
                      seal::CKKSEncoder& ckks_encoder,
                      seal::Encryptor& encryptor, seal::Decryptor& decryptor);

void scalar_relu_seal(const HEType& arg, HEType& out,
                      const HESealBackend& he_seal_backend);

void relu_seal(const std::vector<HEType>& arg, std::vector<HEType>& out,
               size_t count, const HESealBackend& he_seal_backend);

}  // namespace ngraph::he
