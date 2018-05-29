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

#include <unordered_set>

#include "he_seal_parameter.hpp"
#include "ngraph/except.hpp"

using namespace ngraph;
using namespace std;

runtime::he::HESealParameter::HESealParameter(uint64_t poly_modulus, uint64_t plain_modulus)
    : HEParameter(poly_modulus, plain_modulus)
{
}

runtime::he::HESealParameter::HESealParameter(uint64_t poly_modulus,
                                              uint64_t plain_modulus,
                                              uint64_t security_level,
                                              int fractional_encoder_integer_coeff_count,
                                              int fractional_encoder_fraction_coeff_count,
                                              uint64_t fractional_encoder_base,
                                              int evaluation_decomposition_bit_count)
    : HEParameter(poly_modulus, plain_modulus)
    , m_security_level(security_level)
    , m_fractional_encoder_integer_coeff_count(fractional_encoder_integer_coeff_count)
    , m_fractional_encoder_fraction_coeff_count(fractional_encoder_fraction_coeff_count)
    , m_fractional_encoder_base(fractional_encoder_base)
    , m_evaluation_decomposition_bit_count(evaluation_decomposition_bit_count)
{
}
