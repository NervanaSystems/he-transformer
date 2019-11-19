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

#include "seal/kernel/rescale_seal.hpp"

#include <memory>
#include <vector>

namespace ngraph::runtime::he {

void rescale_seal(std::vector<HEType>& arg, HESealBackend& he_seal_backend,
                  const bool verbose) {
  if (verbose) {
    NGRAPH_HE_LOG(3) << "Rescaling " << arg.size() << " elements";
  }

  using Clock = std::chrono::high_resolution_clock;
  auto t1 = Clock::now();
  size_t new_chain_index = std::numeric_limits<size_t>::max();

  bool all_plaintexts = true;
  for (auto& he_type : arg) {
    if (he_type.is_ciphertext()) {
      size_t curr_chain_index =
          he_seal_backend.get_chain_index(*he_type.get_ciphertext());
      if (curr_chain_index == 0) {
        new_chain_index = 0;
      } else {
        new_chain_index = curr_chain_index - 1;
      }
      all_plaintexts = false;
      break;
    }
  }

  if (all_plaintexts) {
    if (verbose) {
      NGRAPH_HE_LOG(3) << "Skipping rescaling because all values are known";
    }
    return;
  }

  NGRAPH_CHECK(new_chain_index != std::numeric_limits<size_t>::max(),
               "Invalid new chain index in rescaling");
  if (new_chain_index == 0) {
    if (verbose) {
      NGRAPH_HE_LOG(3) << "Skipping rescaling to chain index 0";
    }
    return;
  }
  if (verbose) {
    NGRAPH_HE_LOG(3) << "New chain index " << new_chain_index;
  }

#pragma omp parallel for
  for (size_t i = 0; i < arg.size(); ++i) {  // NOLINT
    auto cipher = arg[i];
    if (arg[i].is_ciphertext()) {
      he_seal_backend.get_evaluator()->rescale_to_next_inplace(
          arg[i].get_ciphertext()->ciphertext());
    }
  }
  if (verbose) {
    auto t2 = Clock::now();
    NGRAPH_HE_LOG(3) << "Rescale_xxx took "
                     << std::chrono::duration_cast<std::chrono::milliseconds>(
                            t2 - t1)
                            .count()
                     << "ms";
  }
}

}  // namespace ngraph::runtime::he
