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

#include <iostream>
#include <memory>
#include <vector>

#include "ENCRYPTO_utils/crypto/crypto.h"
#include "aby/aby_util.hpp"
#include "abycore/aby/abyparty.h"
#include "abycore/circuit/booleancircuits.h"
#include "abycore/circuit/share.h"
#include "abycore/sharing/sharing.h"
#include "logging/ngraph_he_log.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"

namespace ngraph::runtime::aby {
// @param xs: server share of X, values in [0,q]
// @param xc: client share of X, values in [0,q]
// @param rs: server share of output random mask, values in [0,q]
// @param bounds: server share of bounded relu value
// @param coeff_modulus: q
// @brief Let x = (xs+xc)mod q; Then, the circuit returns
//    max_i x_i + r
inline share* maxpool_aby(BooleanCircuit& circ, size_t num_input_vals,
                          std::vector<uint64_t>& xs, std::vector<uint64_t>& xc,
                          uint64_t r, size_t bitlen, size_t coeff_modulus) {
  size_t q = coeff_modulus;
  size_t q_half = coeff_modulus / 2;
  NGRAPH_HE_LOG(3) << "Creating ABY Maxpool circuit with " << num_input_vals
                   << " values, q = " << q << ", q/2 " << q_half;

  NGRAPH_CHECK(xs.size() == num_input_vals, "Wrong number of xs (got ",
               xs.size(), ", expected ", num_input_vals, ")");
  NGRAPH_CHECK(xc.size() == num_input_vals, "Wrong number of xc (got ",
               xc.size(), ", expected ", num_input_vals, ")");
  check_argument_range(xs, 0UL, coeff_modulus);
  check_argument_range(xc, 0UL, coeff_modulus);
  check_argument_range(std::vector<uint64_t>{r}, 0UL, coeff_modulus);

  print_argument(xs, "xs");
  print_argument(xc, "xc");
  print_argument(std::vector<uint64_t>{r}, "r");

  share* out;
  share* xs_in = circ.PutSIMDINGate(num_input_vals, xs.data(), bitlen, SERVER);
  share* xc_in = circ.PutSIMDINGate(num_input_vals, xc.data(), bitlen, CLIENT);
  share* r_in = circ.PutSIMDINGate(1, &r, bitlen, SERVER);

  share* Q = circ.PutSIMDCONSGate(num_input_vals, q, bitlen);

  // Reconstruct input x = (xs + xc) mod q
  share* x = circ.PutADDGate(xs_in, xc_in);
  x = reduce_mod(circ, x, Q);

  // Compute max
  uint32_t max_idx = 0;
  share* max_x = circ.PutSubsetGate(x, &max_idx, 1);

  for (uint32_t input_idx = 1; input_idx < num_input_vals; input_idx++) {
    NGRAPH_HE_LOG(5) << "input idx " << input_idx;

    share* x_at_idx = circ.PutSubsetGate(x, &input_idx, 1);
    share* x_at_idx_biggest = circ.PutGTGate(x_at_idx, max_x);
    max_x = circ.PutMUXGate(x_at_idx, max_x, x_at_idx_biggest);
  }

  // Additively mask output
  max_x = circ.PutADDGate(max_x, r_in);
  max_x = reduce_mod(circ, max_x, Q);

  out = circ.PutOUTGate(max_x, CLIENT);
  return out;
}

}  // namespace ngraph::runtime::aby