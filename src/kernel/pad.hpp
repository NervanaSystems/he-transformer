//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <cmath>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            class HEBackend;
            class HECiphertext;
            class HEPlaintext;

            namespace kernel
            {
                void pad(
                    // Input tensor, must be ciphertext
                    const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& arg0,
                    // Padding value, must be scalar, plaintext
                    const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& arg1,
                    // Output tensor must be ciphertext
                    const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& out,
                    const Shape& arg0_shape,
                    const Shape& out_shape,
                    const Shape& padding_below,
                    const Shape& padding_above,
                    const Shape& padding_interior,
                    const std::shared_ptr<runtime::he::HEBackend>& he_backend);
            }
        }
    }
}
