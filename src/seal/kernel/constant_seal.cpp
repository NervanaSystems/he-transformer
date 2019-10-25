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

#include "seal/kernel/constant_seal.hpp"
#include "seal/seal_util.hpp"
#include "util.hpp"

namespace ngraph {
namespace he {

void constant_seal(std::vector<HEType>& out, const element::Type& element_type,
                   const void* data_ptr, const HESealBackend& he_seal_backend,
                   size_t count) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
  size_t type_byte_size = element_type.size();
  if (out.size() != count) {
    throw ngraph_error("out.size() != count for constant op");
  }

#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    const void* src = static_cast<const char*>(data_ptr) + i * type_byte_size;
    auto plaintext =
        HEPlaintext(std::vector<double>{type_to_double(src, element_type)});
    if (out[i].is_plaintext()) {
      out[i].set_plaintext(plaintext);
    } else {
      he_seal_backend.encrypt(out[i].get_ciphertext(), plaintext, element_type,
                              he_seal_backend.complex_packing());
    }
  }
}

}  // namespace he
}  // namespace ngraph