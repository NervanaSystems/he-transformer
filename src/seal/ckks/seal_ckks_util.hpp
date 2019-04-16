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
// Matches the scale and modulus chain for the elements in-place
void match_modulus_inplace(
    std::vector<std::shared_ptr<runtime::he::HECiphertext>>& elements,
    const runtime::he::he_seal::HESealCKKSBackend* he_seal_ckks_backend);

// Matches the scale and modulus chain for the two elements in-place
// The elements are modified if necessary
template <typename S, typename T>
void match_modulus_inplace(S* arg0, T* arg1,
                           const HESealCKKSBackend* he_seal_ckks_backend) {
  auto scale0 = arg0->get_hetext().scale();
  auto scale1 = arg1->get_hetext().scale();

  if (scale0 < 0.99 * scale1 || scale0 > 1.01 * scale1) {
    NGRAPH_DEBUG << "Scale " << std::setw(10) << scale0
                 << " does not match scale " << scale1
                 << " in scalar add, ratio is " << scale0 / scale1;
  }
  if (scale0 != scale1) {
    arg0->get_hetext().scale() = arg1->get_hetext().scale();
  }

  size_t chain_ind0 = he_seal_ckks_backend->get_context()
                          ->context_data(arg0->get_hetext().parms_id())
                          ->chain_index();

  size_t chain_ind1 = he_seal_ckks_backend->get_context()
                          ->context_data(arg1->get_hetext().parms_id())
                          ->chain_index();

  if (chain_ind0 > chain_ind1) {
    he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
        arg0->get_hetext(), arg1->get_hetext().parms_id());
    chain_ind0 = he_seal_ckks_backend->get_context()
                     ->context_data(arg0->get_hetext().parms_id())
                     ->chain_index();
    assert(chain_ind0 == chain_ind1);
  } else if (chain_ind1 > chain_ind0) {
    he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
        arg1->get_hetext(), arg0->get_hetext().parms_id());
    chain_ind1 = he_seal_ckks_backend->get_context()
                     ->context_data(arg1->get_hetext().parms_id())
                     ->chain_index();
    assert(chain_ind0 == chain_ind1);
  }
}
}  // namespace ckks
}  // namespace he_seal
}  // namespace he
}  // namespace runtime
}  // namespace ngraph