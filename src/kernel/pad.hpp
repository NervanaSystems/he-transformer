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
#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"

namespace ngraph {
namespace runtime {
namespace he {
namespace kernel {
void pad(const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& arg0,
         const std::vector<std::shared_ptr<runtime::he::HECiphertext>>&
             arg1,  // scalar
         std::vector<std::shared_ptr<runtime::he::HECiphertext>>& out,
         const Shape& arg0_shape, const Shape& out_shape,
         const Shape& padding_below, const Shape& padding_above,
         const Shape& padding_interior, size_t batch_size,
         const runtime::he::HEBackend* he_backend);

void pad(const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& arg0,
         const std::vector<std::shared_ptr<runtime::he::HEPlaintext>>&
             arg1,  // scalar
         std::vector<std::shared_ptr<runtime::he::HECiphertext>>& out,
         const Shape& arg0_shape, const Shape& out_shape,
         const Shape& padding_below, const Shape& padding_above,
         const Shape& padding_interior, size_t batch_size,
         const runtime::he::HEBackend* he_backend);
}  // namespace kernel
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
