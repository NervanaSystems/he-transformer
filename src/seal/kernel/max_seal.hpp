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

#include <vector>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"

namespace ngraph::he {

inline void max_seal(const std::vector<HEType>& arg, std::vector<HEType>& out,
                     const Shape& in_shape, const Shape& out_shape,
                     const AxisSet& reduction_axes, size_t batch_size,
                     const seal::parms_id_type& parms_id, double scale,
                     seal::CKKSEncoder& ckks_encoder,
                     seal::Encryptor& encryptor, seal::Decryptor& decryptor) {
  std::vector<HEPlaintext> out_plain(
      out.size(), HEPlaintext(std::vector<double>(
                      batch_size, -std::numeric_limits<double>::infinity())));

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
      decrypt(max_cmp_plain, *max_cmp.get_ciphertext(),
              max_cmp.complex_packing(), decryptor, ckks_encoder);
      max_cmp_plain.resize(batch_size);
    }
    for (size_t i = 0; i < max_cmp_plain.size(); ++i) {
      out_plain[out_idx][i] = std::max(out_plain[out_idx][i], max_cmp_plain[i]);
    }
  }

  for (const Coordinate& output_coord : output_transform) {
    size_t out_idx = output_transform.index(output_coord);
    if (out[out_idx].is_plaintext()) {
      out[out_idx].set_plaintext(out_plain[out_idx]);
    } else {
      encrypt(out[out_idx].get_ciphertext(), out_plain[out_idx], parms_id,
              ngraph::element::f32, scale, ckks_encoder, encryptor,
              out[out_idx].complex_packing());
    }
  }
}

inline void max_seal(const std::vector<HEType>& arg, std::vector<HEType>& out,
                     const Shape& in_shape, const Shape& out_shape,
                     const AxisSet& reduction_axes, size_t batch_size,
                     const HESealBackend& he_seal_backend) {
  max_seal(arg, out, in_shape, out_shape, reduction_axes, batch_size,
           he_seal_backend.get_context()->first_parms_id(),
           he_seal_backend.get_scale(), *he_seal_backend.get_ckks_encoder(),
           *he_seal_backend.get_encryptor(), *he_seal_backend.get_decryptor());
}

}  // namespace ngraph::he
