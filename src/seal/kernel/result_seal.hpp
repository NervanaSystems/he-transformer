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

#include "he_plaintext.hpp"
#include "ngraph/check.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {

inline void scalar_result_seal(const HEType& arg, HEType& out,
                               HESealBackend& he_seal_backend) {
  out.complex_packing() = arg.complex_packing();

  if (arg.is_ciphertext() && out.is_ciphertext()) {
    out = arg;
  } else if (arg.is_ciphertext() && out.is_plaintext()) {
    // TODO(fboemer): decrypt instead?
    out.set_ciphertext(arg.get_ciphertext());
  } else if (arg.is_plaintext() && out.is_ciphertext()) {
    // TODO(fboemer): encrypt instead?
    out.set_plaintext(arg.get_plaintext());

  } else if (arg.is_plaintext() && out.is_plaintext()) {
    out = arg;
  } else {
    NGRAPH_CHECK(false, "Unknown result type");
  }
}

inline void result_seal(const std::vector<HEType>& arg,
                        std::vector<HEType>& out, const size_t count,
                        HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(arg.size() >= count, "Result arg size ", arg.size(),
               " smaller than count ", count);
  NGRAPH_CHECK(out.size() >= count, "Result out size ", out.size(),
               " smaller than count ", count);

#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_result_seal(arg[i], out[i], he_seal_backend);
  }
}

}  // namespace he
}  // namespace ngraph
