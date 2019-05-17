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
#include <mutex>

#include "seal/ckks/seal_ckks_util.hpp"

using namespace ngraph;

// Matches the modulus chain for the two elements in-place
// The elements are modified if necessary
void runtime::he::he_seal::ckks::match_modulus_inplace(
    SealPlaintextWrapper* arg0, SealPlaintextWrapper* arg1,
    const HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  if (arg0 == arg1) {
    return;
  }
  std::lock_guard<std::mutex>(arg0->get_encode_mutex());
  std::lock_guard<std::mutex>(arg1->get_encode_mutex());
  size_t chain_ind0 = he_seal_ckks_backend->get_context()
                          ->context_data(arg0->get_hetext().parms_id())
                          ->chain_index();

  size_t chain_ind1 = he_seal_ckks_backend->get_context()
                          ->context_data(arg1->get_hetext().parms_id())
                          ->chain_index();

  if (chain_ind0 != chain_ind1) {
    NGRAPH_INFO << "Chain inds " << chain_ind0 << ", " << chain_ind1
                << " do not match";
  }

  if (chain_ind0 > chain_ind1) {
    he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
        arg0->get_hetext(), arg1->get_hetext().parms_id());
    chain_ind0 = he_seal_ckks_backend->get_context()
                     ->context_data(arg0->get_hetext().parms_id())
                     ->chain_index();
    NGRAPH_ASSERT(chain_ind0 == chain_ind1);
  } else if (chain_ind1 > chain_ind0) {
    he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
        arg1->get_hetext(), arg0->get_hetext().parms_id());
    chain_ind1 = he_seal_ckks_backend->get_context()
                     ->context_data(arg1->get_hetext().parms_id())
                     ->chain_index();
    NGRAPH_ASSERT(chain_ind0 == chain_ind1);
  }
}

// Matches the modulus chain for the two elements in-place
// The elements are modified if necessary
void runtime::he::he_seal::ckks::match_modulus_inplace(
    SealPlaintextWrapper* arg0, SealCiphertextWrapper* arg1,
    const HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  std::lock_guard<std::mutex>(arg0->get_encode_mutex());
  std::lock_guard<std::mutex>(arg1->get_mutex());

  size_t chain_ind0 = he_seal_ckks_backend->get_context()
                          ->context_data(arg0->get_hetext().parms_id())
                          ->chain_index();

  size_t chain_ind1 = he_seal_ckks_backend->get_context()
                          ->context_data(arg1->get_hetext().parms_id())
                          ->chain_index();

  if (chain_ind0 != chain_ind1) {
    NGRAPH_INFO << "Chain inds " << chain_ind0 << ", " << chain_ind1
                << " do not match";
  }

  NGRAPH_ASSERT(chain_ind0 >= chain_ind1)
      << "plaintext chain ind " << chain_ind1
      << " smaller than ciphertext chain in " << chain_ind0;

  if (chain_ind0 > chain_ind1) {
    he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
        arg0->get_hetext(), arg1->get_hetext().parms_id());
    chain_ind0 = he_seal_ckks_backend->get_context()
                     ->context_data(arg0->get_hetext().parms_id())
                     ->chain_index();
    NGRAPH_ASSERT(chain_ind0 == chain_ind1);
  }
}

// Matches the modulus chain for the two elements in-place
// The elements are modified if necessary
void runtime::he::he_seal::ckks::match_modulus_inplace(
    SealCiphertextWrapper* arg0, SealPlaintextWrapper* arg1,
    const HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  runtime::he::he_seal::ckks::match_modulus_inplace(arg1, arg0,
                                                    he_seal_ckks_backend, pool);
}

void runtime::he::he_seal::ckks::match_modulus_inplace(
    SealCiphertextWrapper* arg0, SealCiphertextWrapper* arg1,
    const HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
  if (arg0 == arg1) {
    return;
  }
  std::lock_guard<std::mutex>(arg0->get_mutex());
  std::lock_guard<std::mutex>(arg1->get_mutex());

  size_t chain_ind0 = he_seal_ckks_backend->get_context()
                          ->context_data(arg0->get_hetext().parms_id())
                          ->chain_index();

  size_t chain_ind1 = he_seal_ckks_backend->get_context()
                          ->context_data(arg1->get_hetext().parms_id())
                          ->chain_index();

  if (chain_ind0 != chain_ind1) {
    NGRAPH_INFO << "Chain inds " << chain_ind0 << ", " << chain_ind1
                << " do not match";
  } else {
    NGRAPH_INFO << "chain idx " << chain_ind0;
  }

  if (chain_ind0 > chain_ind1) {
    he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
        arg0->get_hetext(), arg1->get_hetext().parms_id(), pool);
    chain_ind0 = he_seal_ckks_backend->get_context()
                     ->context_data(arg0->get_hetext().parms_id())
                     ->chain_index();
    NGRAPH_ASSERT(chain_ind0 == chain_ind1);
  } else if (chain_ind1 > chain_ind0) {
    he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
        arg1->get_hetext(), arg0->get_hetext().parms_id(), pool);
    chain_ind1 = he_seal_ckks_backend->get_context()
                     ->context_data(arg1->get_hetext().parms_id())
                     ->chain_index();
    NGRAPH_ASSERT(chain_ind0 == chain_ind1);
  }
}