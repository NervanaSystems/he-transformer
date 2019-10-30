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

#include <cmath>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/except.hpp"
#include "seal/kernel/pad_seal.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph {
namespace he {

void pad_seal(std::vector<HEType>& arg0,
              std::vector<HEType>& arg1,  // scalar
              std::vector<HEType>& out, const Shape& arg0_shape,
              const Shape& out_shape, const CoordinateDiff& padding_below,
              const CoordinateDiff& padding_above, op::PadMode pad_mode) {
  if (arg1.size() != 1) {
    throw ngraph_error("Padding element must be scalar");
  }
  auto& pad_val = arg1[0];

  // start at (0,0,...,0)
  Coordinate input_start(arg0_shape.size(), 0);
  // end at (d'0,d'1,...,d'n), the outer corner of the post-padding shape
  Coordinate input_end = out_shape;

  Strides input_strides(arg0_shape.size(), 1);

  AxisVector input_axis_order(arg0_shape.size());
  for (size_t i = 0; i < arg0_shape.size(); i++) {
    input_axis_order[i] = i;
  }

  CoordinateTransform input_transform(arg0_shape, input_start, input_end,
                                      input_strides, input_axis_order,
                                      padding_below, padding_above);
  CoordinateTransform output_transform(out_shape);

  CoordinateTransform::Iterator output_it = output_transform.begin();

  NGRAPH_CHECK(shape_size(input_transform.get_target_shape()) ==
               shape_size(output_transform.get_target_shape()));

  for (const Coordinate& in_coord : input_transform) {
    const Coordinate& out_coord = *output_it;

    HEType v(HEPlaintext(), false);

    switch (pad_mode) {
      case op::PadMode::CONSTANT:
        // If the coordinate is out of bounds, substitute pad_val.
        v = input_transform.has_source_coordinate(in_coord)
                ? arg0[input_transform.index(in_coord)]
                : pad_val;
        break;
      case op::PadMode::EDGE: {
        Coordinate c = in_coord;  // have to copy because in_coord is const

        // Truncate each out-of-bound dimension.
        for (size_t i = 0; i < c.size(); i++) {
          if (static_cast<ptrdiff_t>(c[i]) < padding_below[i]) {
            c[i] = padding_below[i];
          }

          if (static_cast<ptrdiff_t>(c[i]) >=
              (padding_below[i] + static_cast<ptrdiff_t>(arg0_shape[i]))) {
            c[i] = static_cast<size_t>(
                padding_below[i] + static_cast<ptrdiff_t>(arg0_shape[i]) - 1);
          }
        }
        v = arg0[input_transform.index(c)];
        break;
      }
      case op::PadMode::REFLECT: {
        // The algorithm here is a bit complicated because if the padding is
        // bigger than the tensor, we may reflect multiple times.
        //
        // Example:
        //
        // Input shape:     [2]
        // Padding:         6 below, 6 above
        // Output shape:    [14]
        //
        // Input:                       a b
        // Expected output: a b a b a b a b a b a b a b
        //
        // Computation for coordinate 13 of output:
        //
        //         . . . . . . a b . . . . .[.] -> (oob above by 6 spaces, so
        //         reflection is at top-6)
        //         .[.]. . . . a b . . . . . .  -> (oob below by 5 spaces, so
        //         reflection is at bottom+5) . . . . . . a b . . .[.]. .  ->
        //         (oob above by 4 spaces, so reflection is at top-4) . . .[.].
        //         . a b . . . . . .  -> (oob below by 3 spaces, so reflection
        //         is at bottom+3) . . . . . . a b .[.]. . . .  -> (oob above by
        //         2 spaces, so reflection is at top-2) . . . . .[.]a b . . . .
        //         . .  -> (oob below by 1 space,  so reflection is at bottom+1)
        //         . . . . . . a[b]. . . . . .  -> (no longer oob, so copy from
        //         here)
        //
        // Note that this algorithm works because REFLECT padding only makes
        // sense if each dim is >= 2.
        Coordinate c = in_coord;  // have to copy because in_coord is const

        for (size_t i = 0; i < c.size(); i++) {
          ptrdiff_t new_dim = c[i];
          bool done_reflecting = false;

          while (!done_reflecting) {
            if (new_dim < padding_below[i]) {
              ptrdiff_t distance_oob = padding_below[i] - new_dim;
              new_dim = padding_below[i] + distance_oob;
            } else if (new_dim >= padding_below[i] +
                                      static_cast<ptrdiff_t>(arg0_shape[i])) {
              ptrdiff_t distance_oob =
                  new_dim - padding_below[i] -
                  (static_cast<ptrdiff_t>(arg0_shape[i]) - 1);
              new_dim = padding_below[i] +
                        static_cast<ptrdiff_t>(arg0_shape[i]) - distance_oob -
                        1;
            } else {
              done_reflecting = true;
            }
          }

          c[i] = static_cast<size_t>(new_dim);
        }
        v = arg0[input_transform.index(c)];
        break;
      }
      case op::PadMode::SYMMETRIC: {
        // TODO(fboemer): Add support for Symmetric mode
        throw ngraph_error("Symmetric mode padding not supported");
      }
    }

    out[output_transform.index(out_coord)] = v;

    ++output_it;
  }
}
}  // namespace he
}  // namespace ngraph
