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

#include "seal/seal.h"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            struct SEALParameter
            {
                // SEALContext
                // Must be 1024, 2048, 4096, 8192, 16384, or 32768, aka n
                std::uint64_t poly_modulus_degree;
                // Must be 128 or 192
                std::uint64_t security_level;
                std::uint64_t plain_modulus;

                // FractionalEncoder
                int fractional_encoder_integer_coeff_count;
                int fractional_encoder_fraction_coeff_count;
                std::uint64_t fractional_encoder_base;

                // generate_evaluation_keys
                int evaluation_decomposition_bit_count;
            };

            static SEALParameter default_seal_parameter{
                16384, // poly_modulus_degree
                128,   // security_level
                // 10000000000, // plain_modulus
                // 1099511627776 , // plain_modulus (1 << 40)
                //1125899906842624, // (1 << 50)
                //1UL << 52,
                1UL << 50,
                //4611686018427387904 - 1, // (1 << 62 - 1)
                100, // fractional_encoder_integer_coeff_count
                100, // fractional_encoder_fraction_coeff_count
                2,   // fractional_encoder_base
                16   // evaluation_decomposition_bit_count
            };

            void assert_valid_seal_parameter(const SEALParameter& sp);

            std::shared_ptr<seal::SEALContext> make_seal_context(const SEALParameter& sp);
        }
    }
}
