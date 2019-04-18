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
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/ckks/seal_ckks_util.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

using namespace ngraph;

void ngraph::runtime::he::he_seal::ckks::match_modulus_inplace(
    std::vector<std::shared_ptr<runtime::he::HECiphertext>>& elements,
    const runtime::he::he_seal::HESealCKKSBackend* he_seal_ckks_backend) {
  NGRAPH_ASSERT(elements.size() > 0) << "elements.size() == 0)";

  auto min_elem = std::dynamic_pointer_cast<SealCiphertextWrapper>(elements[0]);
  NGRAPH_ASSERT(min_elem != nullptr) << "element is not SealCiphertext";

  double min_elem_scale = min_elem->get_hetext().scale();
  size_t min_chain_index = he_seal_ckks_backend->get_context()
                               ->context_data(min_elem->get_hetext().parms_id())
                               ->chain_index();
  for (const auto& elem : elements) {
    auto seal_cipher = std::dynamic_pointer_cast<SealCiphertextWrapper>(elem);
    NGRAPH_ASSERT(seal_cipher != nullptr) << "element is not SealCiphertext";
    size_t elem_chain_index =
        he_seal_ckks_backend->get_context()
            ->context_data(seal_cipher->get_hetext().parms_id())
            ->chain_index();
    if (elem_chain_index < min_chain_index) {
      min_elem = seal_cipher;
      min_chain_index = elem_chain_index;
      min_elem_scale = seal_cipher->get_hetext().scale();
    }
  }
  for (auto& elem : elements) {
    auto cipher_wrapper =
        std::dynamic_pointer_cast<SealCiphertextWrapper>(elem);
    NGRAPH_ASSERT(cipher_wrapper != nullptr) << "element is not SealCiphertext";
    he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
        cipher_wrapper->get_hetext(), min_elem->get_hetext().parms_id());
    auto current_scale = cipher_wrapper->get_hetext().scale();
    if (current_scale < 0.99 * min_elem_scale ||
        current_scale > 1.01 * min_elem_scale) {
      NGRAPH_DEBUG << "Scale " << std::setw(10) << current_scale
                   << " does not match scale " << min_elem_scale
                   << " in scalar add, ratio is "
                   << current_scale / min_elem_scale;
    }
    cipher_wrapper->get_hetext().scale() = min_elem_scale;

// Matches the modulus chain for the two elements in-place
// The elements are modified if necessary
void runtime::he::he_seal::ckks::match_modulus_inplace(
    SealPlaintextWrapper* arg0, SealPlaintextWrapper* arg1,
    const HESealCKKSBackend* he_seal_ckks_backend,
    const seal::MemoryPoolHandle& pool) {
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
  size_t chain_ind0 = he_seal_ckks_backend->get_context()
                          ->context_data(arg0->get_hetext().parms_id())
                          ->chain_index();

  size_t chain_ind1 = he_seal_ckks_backend->get_context()
                          ->context_data(arg1->get_hetext().parms_id())
                          ->chain_index();

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