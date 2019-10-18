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

#include "seal/kernel/subtract_seal.hpp"
#include "seal/kernel/add_seal.hpp"
#include "seal/kernel/negate_seal.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {

void scalar_subtract_seal(SealCiphertextWrapper& arg0,
                          SealCiphertextWrapper& arg1,
                          std::shared_ptr<SealCiphertextWrapper>& out,
                          HESealBackend& he_seal_backend,
                          const seal::MemoryPoolHandle& pool) {
  match_modulus_and_scale_inplace(arg0, arg1, he_seal_backend, pool);
  he_seal_backend.get_evaluator()->sub(arg0.ciphertext(), arg1.ciphertext(),
                                       out->ciphertext());
}

void scalar_subtract_seal(SealCiphertextWrapper& arg0, const HEPlaintext& arg1,
                          std::shared_ptr<SealCiphertextWrapper>& out,
                          const bool complex_packing,
                          HESealBackend& he_seal_backend) {
  HEPlaintext neg_arg1(arg1.size());
  std::transform(arg1.cbegin(), arg1.cend(), neg_arg1.begin(),
                 std::negate<double>());
  scalar_add_seal(arg0, neg_arg1, out, complex_packing, he_seal_backend);
}

void scalar_subtract_seal(const HEPlaintext& arg0, SealCiphertextWrapper& arg1,
                          std::shared_ptr<SealCiphertextWrapper>& out,
                          const bool complex_packing,
                          HESealBackend& he_seal_backend) {
  auto tmp = HESealBackend::create_empty_ciphertext();
  scalar_negate_seal(arg1, tmp, he_seal_backend);
  scalar_add_seal(arg0, *tmp, out, complex_packing, he_seal_backend);
}

void scalar_subtract_seal(const HEPlaintext& arg0, const HEPlaintext& arg1,
                          HEPlaintext& out) {
  std::vector<double> out_vals(arg0.size());
  std::transform(arg0.begin(), arg0.end(), arg1.begin(), out_vals.begin(),
                 std::minus<double>());

  out = HEPlaintext({out_vals});
}
}  // namespace he
}  // namespace ngraph