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

#include <memory>
#include <vector>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/divide_seal.hpp"
#include "seal/kernel/exp_seal.hpp"
#include "seal/kernel/max_seal.hpp"
#include "seal/kernel/softmax_seal.hpp"
#include "seal/kernel/subtract_seal.hpp"
#include "seal/kernel/sum_seal.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {

void softmax_seal(const std::vector<HEPlaintext>& arg,
                  std::vector<HEPlaintext>& out, const Shape& shape,
                  const AxisSet& axes) {
  auto temp_shape = reduce(shape, axes);
  auto temp_elements = shape_size(temp_shape);
  auto temp_ptr = std::vector<HEPlaintext>(temp_elements);

  max_seal(arg, temp_ptr, shape, temp_shape, axes);

  CoordinateTransform transform(shape);
  CoordinateTransform temp_transform(temp_shape);
  for (const Coordinate& coord : transform) {
    Coordinate temp_coord = reduce(coord, axes);

    scalar_subtract_seal(arg[transform.index(coord)],
                         temp_ptr[temp_transform.index(temp_coord)],
                         out[transform.index(coord)]);
    scalar_exp_seal(out[transform.index(coord)], out[transform.index(coord)]);
  }

  sum_seal(out, temp_ptr, shape, temp_shape, axes);

  for (const Coordinate& coord : transform) {
    Coordinate temp_coord = reduce(coord, axes);

    scalar_divide_seal(out[transform.index(coord)],
                       temp_ptr[temp_transform.index(temp_coord)],
                       out[transform.index(coord)]);
  }
}

void softmax_seal(
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const Shape& shape, const AxisSet& axes,
    const HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(false, "Softmax cipher cipher uniumpleneted");
}

}  // namespace he
}  // namespace ngraph