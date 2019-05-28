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
#include "kernel/negate.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/negate_seal.hpp"

namespace ngraph {
namespace runtime {
namespace he {
namespace kernel {
template <typename T>
void scalar_negate(std::shared_ptr<T>& arg, std::shared_ptr<T>& out,
                   const element::Type& element_type,
                   const ngraph::he::HEBackend* he_backend) {
  auto he_seal_backend = cast_to_seal_backend(he_backend);
  auto arg_seal = cast_to_seal_hetext(arg);
  auto out_seal = cast_to_seal_hetext(out);
  scalar_negate(arg_seal, out_seal, element_type,
                                 he_seal_backend);
}

template <typename T>
void negate(std::vector<std::shared_ptr<T>>& arg,
            std::vector<std::shared_ptr<T>>& out,
            const element::Type& element_type,
            const ngraph::he::HEBackend* he_backend, size_t count) {
#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_negate(arg[i], out[i], element_type, he_backend);
  }
}
}  // namespace kernel
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
