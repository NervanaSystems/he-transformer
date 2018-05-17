/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "ngraph/coordinate_transform.hpp"
#include "he_ciphertext.hpp"
#include "seal/seal.h"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            namespace kernel
            {
                void concat(const vector<vector<shared_ptr<he::HECiphertext>>>& args,
                            vector<shared_ptr<he::HECiphertext>>& out,
                            const std::vector<Shape>& in_shapes,
                            const Shape& out_shape,
                            size_t concatenation_axis);
            }
        }
    }
}
