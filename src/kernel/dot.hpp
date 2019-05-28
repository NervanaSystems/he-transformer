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
namespace runtime {
namespace he {
namespace kernel {
template <typename S, typename T, typename V>
void dot(std::vector<std::shared_ptr<S>>& arg0,
         std::vector<std::shared_ptr<T>>& arg1,
         std::vector<std::shared_ptr<V>>& out, const Shape& arg0_shape,
         const Shape& arg1_shape, const Shape& out_shape,
         size_t reduction_axes_count, const element::Type& element_type,
         const ngraph::he::HEBackend* he_backend);
}
}  // namespace he
}  // namespace runtime
}  // namespace ngraph

template <typename S, typename T, typename V>
void ngraph::ngraph::he::dot(
    std::vector<std::shared_ptr<S>>& arg0,
    std::vector<std::shared_ptr<T>>& arg1, std::vector<std::shared_ptr<V>>& out,
    const Shape& arg0_shape, const Shape& arg1_shape, const Shape& out_shape,
    size_t reduction_axes_count, const element::Type& element_type,
    const ngraph::he::HEBackend* he_backend) {
  auto he_seal_backend = ngraph::he::cast_to_seal_backend(he_backend);
  ngraph::he::dot_seal(
      arg0, arg1, out, arg0_shape, arg1_shape, out_shape, reduction_axes_count,
      element_type, he_seal_backend);
}
