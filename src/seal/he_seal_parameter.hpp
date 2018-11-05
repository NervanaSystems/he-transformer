//*****************************************************************************
// Copyright 2018 Intel Corporation
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
                    struct CoeffModulus;

                    // For BFV
                    HESealParameter(std::string scheme_name,
                                    std::uint64_t poly_modulus_degree,
                                    std::uint64_t plain_modulus,
                                    std::uint64_t security_level,
                                    std::uint64_t evaluation_decomposition_bit_count,
                                    std::uint64_t fractional_encoder_integer_coeff_count,
                                    std::uint64_t fractional_encoder_fraction_coeff_count,
                                    std::uint64_t fractional_encoder_base)
                        : m_scheme_name(scheme_name)
                        , m_poly_modulus_degree(poly_modulus_degree)
                        , m_plain_modulus(plain_modulus)
                        , m_security_level(security_level)
                        , m_evaluation_decomposition_bit_count(evaluation_decomposition_bit_count)
                        , m_fractional_encoder_integer_coeff_count(
                              fractional_encoder_integer_coeff_count)
                        , m_fractional_encoder_fraction_coeff_count(
                              fractional_encoder_fraction_coeff_count)
                        , m_fractional_encoder_base(fractional_encoder_base)
                    {
                    }

                    // For CKKS
                    HESealParameter(std::string scheme_name,
                                    std::uint64_t poly_modulus_degree,
                                    std::uint64_t security_level,
                                    std::uint64_t evaluation_decomposition_bit_count)
                        : m_scheme_name(scheme_name)
                        , m_poly_modulus_degree(poly_modulus_degree)
                        , m_security_level(security_level)
                        , m_evaluation_decomposition_bit_count(evaluation_decomposition_bit_count)
                    {
                    }

                    // For CKKS
                    HESealParameter(std::string scheme_name,
                                    std::uint64_t poly_modulus_degree,
                                    std::uint64_t security_level,
                                    std::uint64_t evaluation_decomposition_bit_count,
                                    CoeffModulus coeff_modulus)
                        : m_scheme_name(scheme_name)
                        , m_poly_modulus_degree(poly_modulus_degree)
                        , m_security_level(security_level)
                        , m_evaluation_decomposition_bit_count(evaluation_decomposition_bit_count)
                        , m_coeff_modulus(coeff_modulus)
                    {
                    }

                    ~HESealParameter() {}
                    // Must be "BFV" or "CKKS"
                    std::string m_scheme_name;

                    // Must be 1024, 2048, 4096, 8192, 16384, or 32768, aka n
                    std::uint64_t m_poly_modulus_degree;
                    std::uint64_t m_plain_modulus;
                    // Must be 128 or 192
                    std::uint64_t m_security_level;

                    // Used to generate relin keys
                    std::uint64_t m_evaluation_decomposition_bit_count;

                    // For CKKS encoder
                    double m_scale;

                    // Size and number of coefficient modulus.
                    typedef struct CoeffModulus
                    {
                        std::uint64_t bit_count;
                        std::uint64_t coeff_count;
                    };
                    CoeffModulus m_coeff_modulus;

                    // For BFV FractionalEncoder
                    std::uint64_t m_fractional_encoder_integer_coeff_count;
                    std::uint64_t m_fractional_encoder_fraction_coeff_count;
                    std::uint64_t m_fractional_encoder_base;
                };
            }
        }
    }
}
