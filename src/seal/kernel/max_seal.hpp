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

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace he {
inline void max_seal(const std::vector<HEPlaintext>& arg,
                     std::vector<HEPlaintext>& out, const Shape& in_shape,
                     const Shape& out_shape, const AxisSet& reduction_axes) {
  size_t batch_size = 1;
  if (arg.size() > 0) {
    batch_size = arg[0].num_values();
    if (batch_size == 0) {
      batch_size = 1;
    }
  }

  HEPlaintext min_val(std::vector<double>(
      batch_size, -std::numeric_limits<double>::infinity()));
  CoordinateTransform output_transform(out_shape);

  for (const Coordinate& output_coord : output_transform) {
    out[output_transform.index(output_coord)] = min_val;
  }

  CoordinateTransform input_transform(in_shape);

  for (const Coordinate& input_coord : input_transform) {
    Coordinate output_coord = reduce(input_coord, reduction_axes);
    size_t out_idx = output_transform.index(output_coord);

    auto x = arg[input_transform.index(input_coord)];
    auto& new_vals = x.values();
    auto& cur_max = out[out_idx].values();

    for (size_t i = 0; i < cur_max.size(); ++i) {
      if (new_vals[i] > cur_max[i]) {
        cur_max[i] = new_vals[i];
      }
    }
  }
}

inline void max_seal(
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const Shape& in_shape, const Shape& out_shape,
    const AxisSet& reduction_axes, const HESealBackend& he_seal_backend) {
  std::vector<HEPlaintext> plain_arg(arg.size());
  std::vector<HEPlaintext> plain_out(out.size());
  for (size_t arg_idx = 0; arg_idx < arg.size(); ++arg_idx) {
    he_seal_backend.decrypt(plain_arg[arg_idx], *arg[arg_idx]);
  }

  max_seal(plain_arg, plain_out, in_shape, out_shape, reduction_axes);

  for (size_t out_idx = 0; out_idx < out.size(); ++out_idx) {
    he_seal_backend.encrypt(out[out_idx], plain_out[out_idx], element::f32,
                            he_seal_backend.complex_packing());
  }
}

}  // namespace he
}  // namespace ngraph
