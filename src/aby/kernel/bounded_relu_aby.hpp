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
#include "aby/util.hpp"
#include "abycore/aby/abyparty.h"
#include "abycore/circuit/booleancircuits.h"
#include "abycore/circuit/share.h"
#include "abycore/sharing/sharing.h"
#include "logging/ngraph_he_log.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"

namespace ngraph {
namespace aby {
// @param xs: server share of X, values in [0,q]
// @param xc: client share of X, values in [0,q]
// @param rs: server share of output random mask, values in [0,q]
// @param bounds: server share of bounded relu value
// @param coeff_modulus: q
// @brief Let x = (xs+xc)mod q; Then, the circuit returns
//    rs              if x < q/2
//   (x + rs) mod q   if x >= q/2
inline share* bounded_relu_aby(BooleanCircuit& circ, size_t num_vals,
                               std::vector<uint64_t>& xs,
                               std::vector<uint64_t>& xc,
                               std::vector<uint64_t>& r,
                               std::vector<uint64_t>& bounds, size_t bitlen,
                               size_t coeff_modulus) {
  NGRAPH_CHECK(xs.size() == num_vals, "Wrong number of xs (got ", xs.size(),
               ", expected ", num_vals, ")");
  NGRAPH_CHECK(xc.size() == num_vals, "Wrong number of xc (got ", xc.size(),
               ", expected ", num_vals, ")");
  NGRAPH_CHECK(r.size() == num_vals, "Wrong number of r (got ", r.size(),
               ", expected ", num_vals, ")");
  NGRAPH_CHECK(bounds.size() == num_vals, "Wrong number of bounds (got ",
               bounds.size(), ", expected ", num_vals, ")");

  size_t q = coeff_modulus;
  size_t q_half = coeff_modulus / 2;
  NGRAPH_HE_LOG(3) << "Creating new bounded relu aby circuit with q = " << q
                   << ", q/2 = " << q_half;

  check_argument_range(xs, 0UL, coeff_modulus);
  check_argument_range(xc, 0UL, coeff_modulus);
  check_argument_range(r, 0UL, coeff_modulus);
  check_argument_range(bounds, 0UL, coeff_modulus);
  print_argument(xs, "xs");
  print_argument(xc, "xc");
  print_argument(r, "r");
  print_argument(bounds, "bounds");

  share* out;
  share* xs_in = circ.PutSIMDINGate(num_vals, xs.data(), bitlen, SERVER);
  share* xc_in = circ.PutSIMDINGate(num_vals, xc.data(), bitlen, CLIENT);
  share* r_in = circ.PutSIMDINGate(num_vals, r.data(), bitlen, SERVER);
  share* bounds_in =
      circ.PutSIMDINGate(num_vals, bounds.data(), bitlen, SERVER);

  share* Q = circ.PutSIMDCONSGate(num_vals, q, bitlen);
  share* zero = circ.PutSIMDCONSGate(num_vals, (size_t)0, bitlen);
  share* half_Q = circ.PutSIMDCONSGate(num_vals, q_half, bitlen);

  // Reconstruct input x = (xs + xc) mod q
  share* x = circ.PutADDGate(xs_in, xc_in);
  x = reduce_mod(circ, x, Q);

  // x > q/2         => brelu(x) = 0
  // q/2 > x > 0     => brelu(x) = bound
  // bound > x       => brelu(x) = x
  share* x_negative = circ.PutGTGate(x, half_Q);
  x = circ.PutMUXGate(zero, x, x_negative);
  share* x_less_than_bound = circ.PutGTGate(bounds_in, x);
  x = circ.PutMUXGate(x, bounds_in, x_less_than_bound);

  // Additively mask output
  x = circ.PutADDGate(x, r_in);
  x = reduce_mod(circ, x, Q);

  out = circ.PutOUTGate(x, CLIENT);

  NGRAPH_INFO << "Creating BRelu circuit";
  return out;
}

}  // namespace aby
}  // namespace ngraph