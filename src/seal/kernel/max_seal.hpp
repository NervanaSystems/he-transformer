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
inline void max_seal(const std::vector<HEType>& arg, std::vector<HEType>& out,
                     const Shape& in_shape, const Shape& out_shape,
                     const AxisSet& reduction_axes, size_t batch_size,
                     const HESealBackend& he_seal_backend) {
  NGRAPH_INFO << "max seal batch size " << batch_size;
  // TODO: use constructor?
  std::vector<HEPlaintext> out_plain(out.size());
  for (size_t i = 0; i < out.size(); ++i) {
    out_plain[i] = HEPlaintext(std::vector<double>(
        batch_size, -std::numeric_limits<double>::infinity()));
  }
  NGRAPH_INFO << "out_plain:";
  for (const auto& elem : out_plain) {
    NGRAPH_INFO << elem;
  }

  CoordinateTransform output_transform(out_shape);
  CoordinateTransform input_transform(in_shape);

  for (const Coordinate& input_coord : input_transform) {
    Coordinate output_coord = reduce(input_coord, reduction_axes);
    size_t out_idx = output_transform.index(output_coord);

    const HEType& max_cmp = arg[input_transform.index(input_coord)];
    HEPlaintext max_cmp_plain;
    if (max_cmp.is_plaintext()) {
      max_cmp_plain = max_cmp.get_plaintext();
    } else {
      he_seal_backend.decrypt(max_cmp_plain, *max_cmp.get_ciphertext(),
                              max_cmp.complex_packing());
      max_cmp_plain.resize(batch_size);
    }
    NGRAPH_INFO << "out_idx " << out_idx;
    NGRAPH_INFO << "Max " << max_cmp_plain << ", " << out_plain[out_idx];

    for (size_t i = 0; i < max_cmp_plain.size(); ++i) {
      out_plain[out_idx][i] = std::max(out_plain[out_idx][i], max_cmp_plain[i]);
    }
    NGRAPH_INFO << " => " << out_plain[out_idx];
  }

  for (const Coordinate& output_coord : output_transform) {
    size_t out_idx = output_transform.index(output_coord);
    if (out[out_idx].is_plaintext()) {
      NGRAPH_INFO << "Setting output " << out_plain[out_idx];
      out[out_idx].set_plaintext(out_plain[out_idx]);
    } else {
      he_seal_backend.encrypt(out[out_idx].get_ciphertext(), out_plain[out_idx],
                              element::f32, out[out_idx].complex_packing());
    }
  }
}

}  // namespace he
}  // namespace ngraph
