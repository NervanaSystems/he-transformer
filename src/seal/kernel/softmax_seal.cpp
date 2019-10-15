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

#include <memory>
#include <vector>

#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/softmax_seal.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {

void softmax_seal(const std::vector<HEPlaintext>& arg,
                  std::vector<HEPlaintext>& out, const Shape& shape,
                  const AxisSet& axes) {
  NGRAPH_CHECK(false, "Softmax plain plain uniumpleneted");
}

void softmax_seal(
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const Shape& shape, const AxisSet& axes,
    const HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(false, "Softmax cipher cipher uniumpleneted");
}

}  // namespace he
}  // namespace ngraph