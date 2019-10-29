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
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {

void softmax_seal(std::vector<HEType>& arg, std::vector<HEType>& out,
                  const Shape& shape, const AxisSet& axes,
                  const element::Type& element_type,
                  HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);

  auto temp_shape = reduce(shape, axes);
  auto temp_elements = shape_size(temp_shape);
  NGRAPH_CHECK(arg.size() > 0, "arg.size() == 0 in softmax");

  // Avoid extra decryption by setting output of max to plaintext
  // TODO(fboemer): avoid extra decryptions in subtract, exp, sum, divide ops below
  auto temp_ptr = std::vector<HEType>(
      temp_elements,
      HEType(HEPlaintext(arg[0].batch_size()), arg[0].complex_packing()));

  max_seal(arg, temp_ptr, shape, temp_shape, axes, arg[0].batch_size(),
           he_seal_backend);

  CoordinateTransform transform(shape);
  CoordinateTransform temp_transform(temp_shape);
  for (const Coordinate& coord : transform) {
    Coordinate temp_coord = reduce(coord, axes);

    scalar_subtract_seal(arg[transform.index(coord)],
                         temp_ptr[temp_transform.index(temp_coord)],
                         out[transform.index(coord)], he_seal_backend);
    scalar_exp_seal(out[transform.index(coord)], out[transform.index(coord)],
                    he_seal_backend);
  }

  sum_seal(out, temp_ptr, shape, temp_shape, axes, element_type,
           he_seal_backend);

  for (const Coordinate& coord : transform) {
    Coordinate temp_coord = reduce(coord, axes);

    scalar_divide_seal(out[transform.index(coord)],
                       temp_ptr[temp_transform.index(temp_coord)],
                       out[transform.index(coord)], he_seal_backend);
  }
}

}  // namespace he
}  // namespace ngraph
