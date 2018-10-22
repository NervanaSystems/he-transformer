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

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "ngraph/log.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            class HEHeaanParameter
            {
            public:
                HEHeaanParameter(std::uint64_t log2_poly_modulus,
                                 std::uint64_t log2_plain_modulus,
                                 std::uint64_t log2_precision)
                    : m_log2_poly_modulus(log2_poly_modulus)
                    , m_log2_plain_modulus(log2_plain_modulus)
                    , m_log2_precision(log2_precision)
                {
                }
                ~HEHeaanParameter() {}
                std::uint64_t m_log2_poly_modulus;
                std::uint64_t m_log2_plain_modulus;
                std::uint64_t m_log2_precision;
            };
        }
    }
}
