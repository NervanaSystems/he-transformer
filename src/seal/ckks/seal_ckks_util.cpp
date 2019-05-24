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

#include "seal/ckks/seal_ckks_util.hpp"

using namespace ngraph;

// Matches the modulus chain for the two elements in-place
// The elements are modified if necessary
void runtime::he::he_seal::ckks::match_modulus_and_scale_inplace(
    SealCiphertextWrapper* arg0, SealCiphertextWrapper* arg1,
    const HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  size_t chain_ind0 =
      runtime::he::he_seal::ckks::get_chain_index(arg0, he_seal_ckks_backend);
  size_t chain_ind1 =
      runtime::he::he_seal::ckks::get_chain_index(arg1, he_seal_ckks_backend);

  if (chain_ind0 == chain_ind1) {
    return;
  }

  if (chain_ind0 < chain_ind1) {
    match_modulus_and_scale_inplace(arg1, arg0, he_seal_ckks_backend, pool);
  }

  bool rescale =
      !runtime::he::he_seal::ckks::within_rescale_tolerance(arg0, arg1);

  if (chain_ind0 > chain_ind1) {
    auto arg1_parms_id = arg1->get_hetext().parms_id();
    if (rescale) {
      he_seal_ckks_backend->get_evaluator()->rescale_to_inplace(
          arg0->get_hetext(), arg1_parms_id);
    } else {
      he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
          arg0->get_hetext(), arg1_parms_id);
    }
    chain_ind0 =
        runtime::he::he_seal::ckks::get_chain_index(arg0, he_seal_ckks_backend);
    NGRAPH_ASSERT(chain_ind0 == chain_ind1);

    runtime::he::he_seal::ckks::match_scale(arg0, arg1, he_seal_ckks_backend);
  }
}