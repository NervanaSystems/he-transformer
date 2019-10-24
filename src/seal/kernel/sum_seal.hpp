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

#include "he_type.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/add_seal.hpp"

namespace ngraph {
namespace he {
inline void sum_seal(std::vector<HEType>& arg, std::vector<HEType>& out,
                     const Shape& in_shape, const Shape& out_shape,
                     const AxisSet& reduction_axes,
                     const element::Type& element_type,
                     HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
  CoordinateTransform output_transform(out_shape);

  bool complex_packing = arg.size() > 0 ? arg[0].complex_packing() : false;
  size_t batch_size = arg.size() > 0 ? arg[0].batch_size() : 1;

  for (const Coordinate& output_coord : output_transform) {
    // TODO: batch size
    const auto out_coord_idx = output_transform.index(output_coord);
    out[out_coord_idx] = HEType(HEPlaintext(std::vector<double>(batch_size, 0)),
                                complex_packing);
  }

  CoordinateTransform input_transform(in_shape);

  for (const Coordinate& input_coord : input_transform) {
    Coordinate output_coord = reduce(input_coord, reduction_axes);

    auto& input = arg[input_transform.index(input_coord)];
    auto& output = out[output_transform.index(output_coord)];
    scalar_add_seal(input, output, output, he_seal_backend);
  }
}

}  // namespace he
}  // namespace ngraph
