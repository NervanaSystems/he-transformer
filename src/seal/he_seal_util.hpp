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

#include <string>

#include "seal/seal.h"

static void print_seal_context(const seal::SEALContext& context)
{
    auto context_data = context.context_data();
    auto scheme_parms = context_data->parms();
    std::string scheme_name =
        (scheme_parms.scheme() == seal::scheme_type::BFV)
            ? "HE:SEAL:BFV"
            : (scheme_parms.scheme() == seal::scheme_type::CKKS) ? "HE:SEAL:CKKS" : "";

    if (scheme_name == "HE:SEAL:BFV")
    {
        NGRAPH_INFO << std::endl
                    << "/ Encryption parameters:" << std::endl
                    << "| scheme: " << scheme_name << std::endl
                    << "| poly_modulus: " << scheme_parms.poly_modulus_degree() << std::endl
                    // Print the size of the true (product) coefficient modulus
                    << "| coeff_modulus size: "
                    << context_data->total_coeff_modulus().significant_bit_count() << " bits"
                    << std::endl
                    << "| plain_modulus: " << scheme_parms.plain_modulus().value() << std::endl
                    << "\\ noise_standard_deviation: " << scheme_parms.noise_standard_deviation();
    }
    else if (scheme_name == "HE:SEAL:CKKS")
    {
        NGRAPH_INFO << std::endl
                    << "/ Encryption parameters:" << std::endl
                    << "| scheme: " << scheme_name << std::endl
                    << "| poly_modulus: " << scheme_parms.poly_modulus_degree() << std::endl
                    // Print the size of the true (product) coefficient modulus
                    << "| coeff_modulus size: "
                    << context_data->total_coeff_modulus().significant_bit_count() << " bits"
                    << std::endl
                    << "\\ noise_standard_deviation: " << scheme_parms.noise_standard_deviation();
    }
}