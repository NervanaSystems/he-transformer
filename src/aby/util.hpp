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

#include <ENCRYPTO_utils/crypto/crypto.h>
#include <iostream>
#include "abycore/aby/abyparty.h"
#include "abycore/circuit/booleancircuits.h"
#include "abycore/circuit/share.h"
#include "abycore/sharing/sharing.h"
#include "logging/ngraph_he_log.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"

namespace ngraph {
namespace aby {
template <typename T>
void print_argument(const std::vector<T>& values, const std::string& name) {
  size_t print_size = std::min(values.size(), 10UL);
  for (size_t i = 0; i < print_size; ++i) {
    NGRAPH_HE_LOG(5) << "\t" << name << "[" << i << "] = " << values[i];
  }
}

template <typename T>
void check_argument_range(const std::vector<T>& values, const T min_val,
                          const T max_val) {
  for (size_t i = 0; i < values.size(); ++i) {
    NGRAPH_CHECK(values[i] >= min_val, "Values[", i, "] (", values[i],
                 ") too small (minimum ", min_val, ")");
    NGRAPH_CHECK(values[i] <= max_val, "Values[", i, "] (", values[i],
                 ") too large (maximum ", max_val, ")");
  }
}

// if (x > mod), let x = x - mod
// else, keep x = x
inline share* reduce_mod(BooleanCircuit& circ, share* x, share* mod) {
  share* sel_share = circ.PutGTGate(mod, x);
  x = circ.PutMUXGate(x, circ.PutSUBGate(x, mod), sel_share);
  return x;
};
}  // namespace aby
}  // namespace ngraph