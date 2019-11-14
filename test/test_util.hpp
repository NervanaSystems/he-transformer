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
//*****************************************************************************

#pragma once

#include <complex>
#include <string>
#include <vector>

#include "he_op_annotations.hpp"
#include "he_tensor.hpp"
#include "logging/ngraph_he_log.hpp"
#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"
#include "seal/he_seal_backend.hpp"

namespace ngraph::he::test {

template <typename T>
bool all_close(const std::vector<std::complex<T>>& a,
               const std::vector<std::complex<T>>& b,
               T atol = static_cast<T>(1e-5)) {
  for (size_t i = 0; i < a.size(); ++i) {
    if ((std::abs(a[i].real() - b[i].real()) > atol) ||
        std::abs(a[i].imag() - b[i].imag()) > atol) {
      NGRAPH_INFO << a[i] << " is not close to " << b[i] << " at index " << i;
      return false;
    }
  }
  return true;
}

template <typename T>
bool all_close(const std::vector<T>& a, const std::vector<T>& b,
               T atol = static_cast<T>(1e-3)) {
  bool close = true;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > atol) {
      NGRAPH_INFO << a[i] << " is not close to " << b[i] << " at index " << i;
      close = false;
    }
  }
  return close;
}

inline std::shared_ptr<HEOpAnnotations> annotation_from_flags(
    const bool from_client, const bool encrypted, const bool packed) {
  return std::make_shared<HEOpAnnotations>(from_client, encrypted, packed);
};

inline std::string config_from_annotation(const HEOpAnnotations& annotation) {
  std::vector<std::string> configs;
  if (annotation.from_client()) {
    configs.emplace_back("from_client");
  }
  if (annotation.encrypted()) {
    configs.emplace_back("encrypt");
  }
  if (annotation.packed()) {
    configs.emplace_back("packed");
  }
  return join(configs, ",");
}

inline std::string config_from_flags(const bool from_client,
                                     const bool encrypted, const bool packed) {
  return config_from_annotation(
      HEOpAnnotations(from_client, encrypted, packed));
}

inline std::shared_ptr<runtime::Tensor> tensor_from_flags(
    HESealBackend& he_seal_backend, const Shape& shape, const bool encrypted,
    const bool packed) {
  if (encrypted && packed) {
    return he_seal_backend.create_packed_cipher_tensor(element::f32, shape);
  }
  if (encrypted && !packed) {
    return he_seal_backend.create_cipher_tensor(element::f32, shape);
  }
  if (!encrypted && packed) {
    return he_seal_backend.create_packed_plain_tensor(element::f32, shape);
  }
  if (!encrypted && !packed) {
    return he_seal_backend.create_plain_tensor(element::f32, shape);
  }
  throw ngraph_error("Logic error");
};

}  // namespace ngraph::he::test