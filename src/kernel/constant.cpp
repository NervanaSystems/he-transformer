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

#include "kernel/constant.hpp"

void ngraph::he::constant(
    std::vector<std::shared_ptr<ngraph::he::HEPlaintext>>& out,
    const element::Type& element_type, const void* data_ptr,
    const ngraph::he::HEBackend* he_backend, size_t count) {
  NGRAPH_CHECK(element_type == element::f32, "Constant supports only f32 type");
  size_t type_byte_size = element_type.size();
  if (out.size() != count) {
    throw ngraph_error("out.size() != count for constant op");
  }

#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    const void* src_with_offset = (void*)((char*)data_ptr + i * type_byte_size);
    float f = *(float*)src_with_offset;
    out[i]->set_values({f});
    // Avoid encoding here (just-in-time encoding)
  }
}

void ngraph::he::constant(
    std::vector<std::shared_ptr<ngraph::he::HECiphertext>>& out,
    const element::Type& element_type, const void* data_ptr,
    const ngraph::he::HEBackend* he_backend, size_t count) {
  NGRAPH_CHECK(element_type == element::f32, "Constant supports only f32 type");
  size_t type_byte_size = element_type.size();
  if (out.size() != count) {
    throw ngraph_error("out.size() != count for constant op");
  }

#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    const void* src_with_offset = (void*)((char*)data_ptr + i * type_byte_size);
    std::unique_ptr<ngraph::he::HEPlaintext> plaintext =
        he_backend->create_empty_plaintext();
    he_backend->encode(*plaintext, src_with_offset, element_type);
    he_backend->encrypt(out[i], *plaintext);
  }
}
