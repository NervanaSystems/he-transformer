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

#include "seal/kernel/add_seal.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {

void scalar_add_seal(SealCiphertextWrapper& arg0, SealCiphertextWrapper& arg1,
                     std::shared_ptr<SealCiphertextWrapper>& out,
                     HESealBackend& he_seal_backend,
                     const seal::MemoryPoolHandle& pool) {
  match_modulus_and_scale_inplace(arg0, arg1, he_seal_backend, pool);
  he_seal_backend.get_evaluator()->add(arg0.ciphertext(), arg1.ciphertext(),
                                       out->ciphertext());
}

void scalar_add_seal(SealCiphertextWrapper& arg0, const HEPlaintext& arg1,
                     std::shared_ptr<SealCiphertextWrapper>& out,
                     const bool complex_packing,
                     HESealBackend& he_seal_backend) {
  // TODO(fboemer): handle case where arg1 = {0, 0, 0, 0, ...}
  bool add_zero = (arg1.size() == 1) && (arg1[0] == 0.0);

  if (add_zero) {
    SealCiphertextWrapper tmp(arg0);
    out = std::make_shared<SealCiphertextWrapper>(tmp);
  } else {
    // TODO(fboemer): optimize for adding single complex number
    if ((arg1.size() == 1) && !complex_packing) {
      add_plain(arg0.ciphertext(), arg1[0], out->ciphertext(), he_seal_backend);
    } else {
      auto p = SealPlaintextWrapper(complex_packing);
      encode(p, arg1, *he_seal_backend.get_ckks_encoder(),
             arg0.ciphertext().parms_id(), element::f32,
             arg0.ciphertext().scale(), complex_packing);
      size_t chain_ind0 = he_seal_backend.get_chain_index(arg0);
      size_t chain_ind1 = he_seal_backend.get_chain_index(p.plaintext());
      NGRAPH_CHECK(chain_ind0 == chain_ind1, "Chain inds ", chain_ind0, ",  ",
                   chain_ind1, " don't match");

      he_seal_backend.get_evaluator()->add_plain(
          arg0.ciphertext(), p.plaintext(), out->ciphertext());
    }
  }
}

void scalar_add_seal(const HEPlaintext& arg0, const HEPlaintext& arg1,
                     HEPlaintext& out) {
  HEPlaintext out_vals;
  if (arg0.size() == 1) {
    std::transform(arg1.begin(), arg1.end(), std::back_inserter(out_vals),
                   std::bind(std::plus<>(), std::placeholders::_1, arg0[0]));
  } else if (arg1.size() == 1) {
    std::transform(arg0.begin(), arg0.end(), std::back_inserter(out_vals),
                   std::bind(std::plus<>(), std::placeholders::_1, arg1[0]));
  } else {
    size_t min_size = std::min(arg0.size(), arg1.size());
    out_vals.resize(min_size);
    for (size_t i = 0; i < min_size; ++i) {
      out_vals[i] = arg0[i] + arg1[i];
    }
  }
  out = std::move(out_vals);
}
}  // namespace he
}  // namespace ngraph
