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
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/add_seal.hpp"

namespace ngraph {
namespace he {
inline void sum_seal(std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg,
                     std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
                     const Shape& in_shape, const Shape& out_shape,
                     const AxisSet& reduction_axes,
                     const element::Type& element_type,
                     ngraph::he::HESealBackend& he_seal_backend) {
  CoordinateTransform output_transform(out_shape);

  for (const Coordinate& output_coord : output_transform) {
    const auto out_coord_idx = output_transform.index(output_coord);
    out[out_coord_idx] = std::make_shared<SealCiphertextWrapper>();
    out[out_coord_idx]->known_value() = true;
    out[out_coord_idx]->value() = 0;
  }

  CoordinateTransform input_transform(in_shape);

  for (const Coordinate& input_coord : input_transform) {
    Coordinate output_coord = reduce(input_coord, reduction_axes);

    auto& input = arg[input_transform.index(input_coord)];
    auto& output = out[output_transform.index(output_coord)];
    ngraph::he::scalar_add_seal(*input, *output, output, element_type,
                                he_seal_backend);
  }
}

inline void sum_seal(std::vector<HEPlaintext>& arg,
                     std::vector<HEPlaintext>& out, const Shape& in_shape,
                     const Shape& out_shape, const AxisSet& reduction_axes,
                     const element::Type& element_type,
                     ngraph::he::HESealBackend& he_seal_backend) {
  CoordinateTransform output_transform(out_shape);

  for (const Coordinate& output_coord : output_transform) {
    out[output_transform.index(output_coord)] = HEPlaintext(0.f);
  }

  CoordinateTransform input_transform(in_shape);

  for (const Coordinate& input_coord : input_transform) {
    Coordinate output_coord = reduce(input_coord, reduction_axes);
    auto& input = arg[input_transform.index(input_coord)];
    auto& output = out[output_transform.index(output_coord)];

    auto tmp = HEPlaintext(output.values());
    ngraph::he::scalar_add_seal(input, tmp, output, element_type,
                                he_seal_backend);
  }
}
}  // namespace he
}  // namespace ngraph
