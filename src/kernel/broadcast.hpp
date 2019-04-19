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

#include "he_ciphertext.hpp"
#include "he_plaintext.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace runtime {
namespace he {
namespace kernel {
void broadcast(const std::vector<std::shared_ptr<HEPlaintext>>& arg,
               std::vector<std::shared_ptr<HEPlaintext>>& out,
               const Shape& in_shape, const Shape& out_shape,
               const AxisSet& broadcast_axes) {
  CoordinateTransform input_transform(in_shape);
  CoordinateTransform output_transform(out_shape);
  for (const Coordinate& output_coord : output_transform) {
    Coordinate input_coord = reduce(output_coord, broadcast_axes);

    out[output_transform.index(output_coord)] =
        std::make_shared<he_seal::SealPlaintextWrapper>(
            *std::dynamic_pointer_cast<he_seal::SealPlaintextWrapper>(
                arg[input_transform.index(input_coord)]));
  }
};

void broadcast(const std::vector<std::shared_ptr<HECiphertext>>& arg,
               std::vector<std::shared_ptr<HECiphertext>>& out,
               const Shape& in_shape, const Shape& out_shape,
               const AxisSet& broadcast_axes) {
  CoordinateTransform input_transform(in_shape);
  CoordinateTransform output_transform(out_shape);
  for (const Coordinate& output_coord : output_transform) {
    Coordinate input_coord = reduce(output_coord, broadcast_axes);

    out[output_transform.index(output_coord)] =
        std::make_shared<he_seal::SealCiphertextWrapper>(
            *std::dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(
                arg[input_transform.index(input_coord)]));
  }
};
}  // namespace kernel
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
