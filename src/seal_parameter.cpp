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

#include "seal/seal.h"
#include "seal_parameter.hpp"
#include "ngraph/except.hpp"

using namespace ngraph;
using namespace std;

void runtime::he::assert_valid_seal_parameter(const runtime::he::SEALParameter& sp)
{
}


shared_ptr<seal::SEALContext> runtime::he::make_seal_context(const runtime::he::SEALParameter& sp)
{
    assert_valid_seal_parameter(sp);

    seal::EncryptionParameters parms;
    parms.set_poly_modulus("1x^" + to_string(sp.poly_modulus_degree) + " + 1");
    if (sp.security_level == 128)
    {
        parms.set_coeff_modulus(seal::coeff_modulus_128(sp.poly_modulus_degree));
    } else if (sp.security_level == 192)
    {
        parms.set_coeff_modulus(seal::coeff_modulus_192(sp.poly_modulus_degree));
    } else {
        throw ngraph_error("security_level must be 128 or 192");
    }
    parms.set_plain_modulus(sp.plain_modulus);
    return make_shared<seal::SEALContext>(parms);
}
