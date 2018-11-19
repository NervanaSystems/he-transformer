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
void dot(const std::vector<std::shared_ptr<S>>& arg0,
         const std::vector<std::shared_ptr<T>>& arg1,
         std::vector<std::shared_ptr<V>>& out, const Shape& arg0_shape,
         const Shape& arg1_shape, const Shape& out_shape,
         size_t reduction_axes_count, const element::Type& element_type,
         const runtime::he::HEBackend* he_backend);
}
}  // namespace he
}  // namespace runtime
}  // namespace ngraph

template <typename S, typename T, typename V>
void ngraph::runtime::he::kernel::dot(
    const std::vector<std::shared_ptr<S>>& arg0,
    const std::vector<std::shared_ptr<T>>& arg1,
    std::vector<std::shared_ptr<V>>& out, const Shape& arg0_shape,
    const Shape& arg1_shape, const Shape& out_shape,
    size_t reduction_axes_count, const element::Type& element_type,
    const runtime::he::HEBackend* he_backend) {
  // Use optimized SEAL dot if possible
  if (auto he_seal_backend =
          dynamic_cast<const runtime::he::he_seal::HESealBackend*>(
              he_backend)) {
    runtime::he::he_seal::kernel::dot_seal(
        arg0, arg1, out, arg0_shape, arg1_shape, out_shape,
        reduction_axes_count, element_type, he_seal_backend);
    return;
  }

  // Get the sizes of the dot axes. It's easiest to pull them from arg1 because
  // they're
  // right up front.
  Shape dot_axis_sizes(reduction_axes_count);
  std::copy(arg1_shape.begin(), arg1_shape.begin() + reduction_axes_count,
            dot_axis_sizes.begin());

  CoordinateTransform arg0_transform(arg0_shape);
  CoordinateTransform arg1_transform(arg1_shape);
  CoordinateTransform output_transform(out_shape);

  // Create coordinate transforms for arg0 and arg1 that throw away the dotted
  // axes.
  size_t arg0_projected_rank = arg0_shape.size() - reduction_axes_count;
  size_t arg1_projected_rank = arg1_shape.size() - reduction_axes_count;

  Shape arg0_projected_shape(arg0_projected_rank);
  std::copy(arg0_shape.begin(), arg0_shape.begin() + arg0_projected_rank,
            arg0_projected_shape.begin());

  Shape arg1_projected_shape(arg1_projected_rank);
  std::copy(arg1_shape.begin() + reduction_axes_count, arg1_shape.end(),
            arg1_projected_shape.begin());

  CoordinateTransform arg0_projected_transform(arg0_projected_shape);
  CoordinateTransform arg1_projected_transform(arg1_projected_shape);

  // Create a coordinate transform that allows us to iterate over all possible
  // values
  // for the dotted axes.
  CoordinateTransform dot_axes_transform(dot_axis_sizes);

  // Get arg0_projected_size and arg1_projected_size for parallelization
  // and pre-compute coordinates
  std::vector<ngraph::Coordinate> arg0_projected_coords;
  for (const Coordinate& coord : arg0_projected_transform) {
    arg0_projected_coords.emplace_back(coord);
  }

  std::vector<ngraph::Coordinate> arg1_projected_coords;
  for (const Coordinate& coord : arg1_projected_transform) {
    arg1_projected_coords.emplace_back(coord);
  }

  size_t arg0_projected_size = arg0_projected_coords.size();
  size_t arg1_projected_size = arg1_projected_coords.size();
  size_t global_projected_size = arg0_projected_size * arg1_projected_size;

#pragma omp parallel for
  for (size_t global_projected_idx = 0;
       global_projected_idx < global_projected_size; ++global_projected_idx) {
    // Compute outer and inner index
    size_t arg0_projected_idx = global_projected_idx / arg1_projected_size;
    size_t arg1_projected_idx = global_projected_idx % arg1_projected_size;

    const Coordinate& arg0_projected_coord =
        arg0_projected_coords[arg0_projected_idx];
    const Coordinate& arg1_projected_coord =
        arg1_projected_coords[arg1_projected_idx];

    // The output coordinate is just the concatenation of the projected
    // coordinates.
    Coordinate out_coord(arg0_projected_coord.size() +
                         arg1_projected_coord.size());

    auto out_coord_it =
        std::copy(arg0_projected_coord.begin(), arg0_projected_coord.end(),
                  out_coord.begin());
    std::copy(arg1_projected_coord.begin(), arg1_projected_coord.end(),
              out_coord_it);

    size_t out_index = output_transform.index(out_coord);

    // Walk along the dotted axes.
    Coordinate arg0_coord(arg0_shape.size());
    Coordinate arg1_coord(arg1_shape.size());
    auto arg0_it = std::copy(arg0_projected_coord.begin(),
                             arg0_projected_coord.end(), arg0_coord.begin());

    std::shared_ptr<V> sum = he_backend->create_empty_hetext<V>(V{});

    bool first_add = true;

    for (const Coordinate& dot_axis_positions : dot_axes_transform) {
      // In order to find the points to multiply together, we need to inject our
      // current
      // positions along the dotted axes back into the projected arg0 and arg1
      // coordinates.
      std::copy(dot_axis_positions.begin(), dot_axis_positions.end(), arg0_it);

      auto arg1_it = std::copy(dot_axis_positions.begin(),
                               dot_axis_positions.end(), arg1_coord.begin());
      std::copy(arg1_projected_coord.begin(), arg1_projected_coord.end(),
                arg1_it);

      // Multiply and add to the summands.
      auto arg0_text = arg0[arg0_transform.index(arg0_coord)];
      auto arg1_text = arg1[arg1_transform.index(arg1_coord)];

      std::shared_ptr<V> prod = he_backend->create_empty_hetext<V>(V{});
      runtime::he::kernel::scalar_multiply(arg0_text.get(), arg1_text.get(),
                                           prod, element_type, he_backend);
      if (first_add) {
        sum = prod;
        first_add = false;
      } else {
        runtime::he::kernel::scalar_add(sum.get(), prod.get(), sum,
                                        element_type, he_backend);
      }
    }
    // Write the sum back.
    out[out_index] = sum;
  }
}
