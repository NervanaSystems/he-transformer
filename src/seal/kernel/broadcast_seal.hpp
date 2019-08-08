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
#include "ngraph/shape_util.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph {
namespace he {
template <typename T>
void broadcast_seal(const std::vector<T>& arg, std::vector<T>& out,
                    const Shape& in_shape, const Shape& out_shape,
                    const AxisSet& broadcast_axes) {
  CoordinateTransform input_transform(in_shape);
  CoordinateTransform output_transform(out_shape);
  NGRAPH_INFO << "Input shape " << join(in_shape, "x");
  NGRAPH_INFO << "out_shape " << join(out_shape, "x");
  NGRAPH_INFO << "broadcast_axes " << broadcast_axes;

  for (const Coordinate& output_coord : output_transform) {
    Coordinate input_coord = reduce(output_coord, broadcast_axes);

    NGRAPH_INFO << "input coord " << input_coord;
    NGRAPH_INFO << "input index " << input_transform.index(input_coord);
    NGRAPH_INFO << "output_coord " << output_coord;
    NGRAPH_INFO << "output index " << output_transform.index(output_coord);
    NGRAPH_INFO << "out.size() " << out.size();
    NGRAPH_INFO << "arg.size() " << arg.size();

    out[output_transform.index(output_coord)] =
        arg[input_transform.index(input_coord)];
  }
};
}  // namespace he
}  // namespace ngraph
