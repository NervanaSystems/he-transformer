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

#include "ngraph/except.hpp"
#include "seal/seal.h"
#include "he_seal_parameter.hpp"

using namespace ngraph;
using namespace std;

runtime::he::HESealParameter::HESealParameter(uint64_t poly_modulus_degree, uint64_t plain_modulus)
    : HEParameter(poly_modulus_degree, plain_modulus)
{
    assert_valid_seal_parameter();
}

runtime::he::HESealParameter::HESealParameter(uint64_t poly_modulus_degree,
                                             uint64_t plain_modulus,
                                             uint64_t security_level,
                                             int fractional_encoder_integer_coeff_count,
                                             int fractional_encoder_fraction_coeff_count,
                                             uint64_t fractional_encoder_base,
                                             int evaluation_decomposition_bit_count)
    : HEParameter(poly_modulus_degree, plain_modulus)
    , m_security_level(security_level)
    , m_fractional_encoder_integer_coeff_count(fractional_encoder_integer_coeff_count)
    , m_fractional_encoder_fraction_coeff_count(fractional_encoder_fraction_coeff_count)
    , m_fractional_encoder_base(fractional_encoder_base)
    , m_evaluation_decomposition_bit_count(evaluation_decomposition_bit_count)
{
}

void runtime::he::HESealParameter::assert_valid_seal_parameter()
{
    static unordered_set<uint64_t> valid_poly_modulus{1024, 2048, 4096, 8192, 16384, 32768};
    if (valid_poly_modulus.count(m_poly_modulus) == 0)
    {
        throw ngraph_error("m_poly_modulus must be 1024, 2048, 4096, 8192, 16384, 32768");
    }
    if (m_security_level != 128 && m_security_level != 192)
    {
        throw ngraph_error("sp.security_level must be 128, 192");
    }
}

/* shared_ptr<seal::SEALContext> runtime::he::make_seal_context(const runtime::he::HESealParameter& sp)
{
    runtime::he::assert_valid_seal_parameter(sp);

    seal::EncryptionParameters parms;
    parms.set_poly_modulus("1x^" + to_string(sp.poly_modulus_degree) + " + 1");
    if (sp.security_level == 128)
    {
        parms.set_coeff_modulus(seal::coeff_modulus_128(sp.poly_modulus_degree));
    }
    else if (sp.security_level == 192)
    {
        parms.set_coeff_modulus(seal::coeff_modulus_192(sp.poly_modulus_degree));
    }
    else
    {
        throw ngraph_error("sp.security_level must be 128, 192");
    }
    parms.set_plain_modulus(sp.plain_modulus);
    return make_shared<seal::SEALContext>(parms);
} */
