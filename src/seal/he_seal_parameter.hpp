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
#include "nlohmann/json.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            namespace he_seal
            {
                class HESealParameter
                {
                    public:
                        HESealParameter(std::uint64_t poly_modulus,
                                        std::uint64_t plain_modulus,
                                        std::uint64_t security_level,
                                        std::string scheme_name,
                                        int evaluation_decomposition_bit_count)
                            : m_poly_modulus(poly_modulus)
                            , m_plain_modulus(plain_modulus)
                            , m_security_level(security_level)
                            , m_scheme_name(scheme_name)
                            , m_evaluation_decomposition_bit_count(evaluation_decomposition_bit_count)
                        {
                        }

                        ~HESealParameter() {}
                        // SEALContext
                        // Must be 1024, 2048, 4096, 8192, 16384, or 32768, aka n
                        std::uint64_t m_poly_modulus;
                        std::uint64_t m_plain_modulus;
                        // Must be 128 or 192
                        std::uint64_t m_security_level;

                        std::string m_scheme_name; // Must be "BFV" or "CKKS"

                        // Used to generate relin keys
                        int m_evaluation_decomposition_bit_count;
                    };
                }
        }
    }
}
