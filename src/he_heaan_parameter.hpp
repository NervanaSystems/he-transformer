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

#include "he_parameter.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            struct HEParameter;

            struct HEHeaanParameter : public HEParameter
            {
                HEHeaanParameter(std::uint64_t log_poly_modulus,
                                 std::uint64_t log_plain_modulus,
                                 std::uint64_t log_precision = 32);

                std::uint64_t m_log_precision;
            };

            static HEHeaanParameter default_heaan_parameter{
                13,  // log_2(poly_modulus)
                383, // log_2(plain_modulus)
                32   // log_2(precision)
            };
        }
    }
}
