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
namespace kernel {
template <typename S, typename T>
std::pair<std::shared_ptr<S>, std::shared_ptr<T>> match_arguments(
    const std::shared_ptr<const S>& arg0, const std::shared_ptr<const T>& arg1,
    const HESealCKKSBackend* he_seal_ckks_backend) {
  auto arg0_scaled = std::make_shared<S>(*arg0);  // ->get_hetext());
  auto arg1_scaled = std::make_shared<T>(*arg1);  //->get_hetext());

  size_t chain_ind0 = he_seal_ckks_backend->get_context()
                          ->context_data(arg0_scaled->get_hetext().parms_id())
                          ->chain_index();

  size_t chain_ind1 = he_seal_ckks_backend->get_context()
                          ->context_data(arg1_scaled->get_hetext().parms_id())
                          ->chain_index();

  NGRAPH_INFO << "Chain inds " << chain_ind0 << ", " << chain_ind1;

  if (chain_ind0 > chain_ind1) {
    he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
        arg0_scaled->get_hetext(), arg1_scaled->get_hetext().parms_id());
    chain_ind0 = he_seal_ckks_backend->get_context()
                     ->context_data(arg0_scaled->get_hetext().parms_id())
                     ->chain_index();
  } else if (chain_ind1 > chain_ind0) {
    he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
        arg1_scaled->get_hetext(), arg0_scaled->get_hetext().parms_id());
    chain_ind1 = he_seal_ckks_backend->get_context()
                     ->context_data(arg1_scaled->get_hetext().parms_id())
                     ->chain_index();
    assert(chain_ind0 == chain_ind1);
  }

  return std::make_pair(arg0_scaled, arg1_scaled);
}
}  // namespace kernel
}  // namespace ckks
}  // namespace he_seal
}  // namespace he
}  // namespace runtime
}  // namespace ngraph