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

#include "he_backend.hpp"
#include "kernel/add.hpp"
#include "kernel/multiply.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/dot_seal.hpp"

namespace ngraph {
namespace he {
inline void dot(const std::vector<std::shared_ptr<HECiphertext>>& arg0,
                const std::vector<std::shared_ptr<HECiphertext>>& arg1,
                std::vector<std::shared_ptr<HECiphertext>>& out,
                const Shape& arg0_shape, const Shape& arg1_shape,
                const Shape& out_shape, size_t reduction_axes_count,
                const element::Type& element_type,
                const ngraph::he::HEBackend* he_backend) {
  throw ngraph_error("dot unimplemented");
}

inline void dot(const std::vector<std::unique_ptr<HEPlaintext>>& arg0,
                const std::vector<std::unique_ptr<HEPlaintext>>& arg1,
                std::vector<std::unique_ptr<HEPlaintext>>& out,
                const Shape& arg0_shape, const Shape& arg1_shape,
                const Shape& out_shape, size_t reduction_axes_count,
                const element::Type& element_type,
                const ngraph::he::HEBackend* he_backend) {
  throw ngraph_error("dot unimplemented");
}

inline void dot(const std::vector<std::unique_ptr<HEPlaintext>>& arg0,
                const std::vector<std::shared_ptr<HECiphertext>>& arg1,
                std::vector<std::shared_ptr<HECiphertext>>& out,
                const Shape& arg0_shape, const Shape& arg1_shape,
                const Shape& out_shape, size_t reduction_axes_count,
                const element::Type& element_type,
                const ngraph::he::HEBackend* he_backend) {
  throw ngraph_error("dot unimplemented");
}
inline void dot(const std::vector<std::shared_ptr<HECiphertext>>& arg0,
                const std::vector<std::unique_ptr<HEPlaintext>>& arg1,
                std::vector<std::shared_ptr<HECiphertext>>& out,
                const Shape& arg0_shape, const Shape& arg1_shape,
                const Shape& out_shape, size_t reduction_axes_count,
                const element::Type& element_type,
                const ngraph::he::HEBackend* he_backend) {
  throw ngraph_error("dot unimplemented");
}
}  // namespace he
}  // namespace ngraph
