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
namespace kernel {
template <typename S, typename T>
std::pair<std::shared_ptr<S>, std::shared_ptr<T>> match_arguments(
    const S* arg0, const T* arg1,
    const HESealCKKSBackend* he_seal_ckks_backend) {
  auto arg0_scaled = std::make_shared<S>(*arg0);
  auto arg1_scaled = std::make_shared<T>(*arg1);

  auto scale0 = arg0_scaled->get_hetext().scale();
  auto scale1 = arg1_scaled->get_hetext().scale();

  if (scale0 < 0.99 * scale1 || scale0 > 1.01 * scale1) {
    NGRAPH_DEBUG << "Scale " << std::setw(10) << scale0
                 << " does not match scale " << scale1
                 << " in scalar add, ratio is " << scale0 / scale1;
  }
  if (scale0 != scale1) {
    arg0_scaled->get_hetext().scale() = arg1_scaled->get_hetext().scale();
  }

  size_t chain_ind0 = he_seal_ckks_backend->get_context()
                          ->context_data(arg0_scaled->get_hetext().parms_id())
                          ->chain_index();

  size_t chain_ind1 = he_seal_ckks_backend->get_context()
                          ->context_data(arg1_scaled->get_hetext().parms_id())
                          ->chain_index();

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