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

#include <string>

#include "he_parameter.hpp"
#include "seal/seal.h"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            struct HEParameter;

            struct HESealParameter : public HEParameter
            {
                HESealParameter(std::uint64_t poly_modulus, std::uint64_t plain_modulus);

                HESealParameter(std::uint64_t poly_modulus,
                                std::uint64_t plain_modulus,
                                std::uint64_t security_level,
                                int fractional_encoder_integer_coeff_count,
                                int fractional_encoder_fraction_coeff_count,
                                std::uint64_t fractional_encoder_base,
                                int evaluation_decomposition_bit_count);

                // SEALContext
                // Must be 1024, 2048, 4096, 8192, 16384, or 32768, aka n
                //std::uint64_t poly_modulus_degree;
                // Must be 128 or 192
                std::uint64_t m_security_level;
                //std::uint64_t plain_modulus;

                // FractionalEncoder
                int m_fractional_encoder_integer_coeff_count;
                int m_fractional_encoder_fraction_coeff_count;
                std::uint64_t m_fractional_encoder_base;

                // generate_evaluation_keys
                int m_evaluation_decomposition_bit_count;
            };

            static HESealParameter default_seal_parameter{
                4096,   // poly_modulus
                2 << 20, // plain_modulus
                128,    // security_level
                64,     // fractional_encoder_integer_coeff_count
                32,     // fractional_encoder_fraction_coeff_count
                2,      // fractional_encoder_base
                16      // evaluation_decomposition_bit_count
            };
        }
    }
}
