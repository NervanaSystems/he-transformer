//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
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

#include <sstream>

#include "gtest/gtest.h"
#include "seal/he_seal_encryption_parameters.hpp"
#include "seal/seal.h"

using namespace std;
using namespace ngraph;
using namespace ngraph::he;

TEST(encryption_parameters, create) {
  size_t poly_modulus_degree{4096};
  auto seal_encryption_parameters =
      seal::EncryptionParameters(seal::scheme_type::CKKS);

  seal_encryption_parameters.set_poly_modulus_degree(poly_modulus_degree);

  auto coeff_modulus =
      seal::CoeffModulus::Create(poly_modulus_degree, vector<int>{30, 30, 30});

  seal_encryption_parameters.set_coeff_modulus(coeff_modulus);

  // Baseline parameters
  EXPECT_NO_THROW(HESealEncryptionParameters(
      "HE_SEAL", seal_encryption_parameters, 128, 1.23, false));

  // Incorrect scheme name
  EXPECT_ANY_THROW(HESealEncryptionParameters(
      "DUMMY_NAME", seal_encryption_parameters, 128, 1.23, false));

  // Incorrect security level
  EXPECT_ANY_THROW(HESealEncryptionParameters(
      "HE_SEAL", seal_encryption_parameters, 123, 1.23, false));

  // Parameters violate security level
  EXPECT_ANY_THROW(HESealEncryptionParameters(
      "HE_SEAL", seal_encryption_parameters, 192, 1.23, false));

  EXPECT_ANY_THROW(HESealEncryptionParameters(
      "HE_SEAL", seal_encryption_parameters, 256, 1.23, false));

  // No enforced security level
  EXPECT_NO_THROW(HESealEncryptionParameters(
      "HE_SEAL", seal_encryption_parameters, 0, 1.23, false));

  // Attributes set properly
  auto parms = HESealEncryptionParameters("HE_SEAL", seal_encryption_parameters,
                                          128, 1.23, true);
  EXPECT_EQ(parms.complex_packing(), true);
  EXPECT_EQ(parms.scale(), 1.23);
  EXPECT_EQ(parms.security_level(), 128);
}

TEST(encryption_parameters, save) {
  size_t poly_modulus_degree{4096};
  auto seal_encryption_parameters =
      seal::EncryptionParameters(seal::scheme_type::CKKS);

  seal_encryption_parameters.set_poly_modulus_degree(poly_modulus_degree);

  auto coeff_modulus =
      seal::CoeffModulus::Create(poly_modulus_degree, vector<int>{30, 30, 30});

  seal_encryption_parameters.set_coeff_modulus(coeff_modulus);

  auto he_parms = HESealEncryptionParameters(
      "HE_SEAL", seal_encryption_parameters, 128, 1.23, false);

  stringstream ss;
  he_parms.save(ss);

  auto loaded_parms = HESealEncryptionParameters::load(ss);

  EXPECT_EQ(he_parms.scale(), loaded_parms.scale());
  EXPECT_EQ(he_parms.complex_packing(), loaded_parms.complex_packing());
  EXPECT_EQ(he_parms.seal_encryption_parameters(),
            loaded_parms.seal_encryption_parameters());
}

TEST(encryption_parameters, from_string) {
  string config = R"(
    {
        "scheme_name" : "HE_SEAL",
        "poly_modulus_degree" : 2048,
        "security_level" : 128,
        "coeff_modulus" : [54],
        "scale" : 1.23
    })";
  auto he_parms =
      HESealEncryptionParameters::parse_config_or_use_default(config.c_str());

  EXPECT_EQ(he_parms.poly_modulus_degree(), 2048);
  EXPECT_EQ(he_parms.security_level(), 128);
  EXPECT_EQ(he_parms.scale(), 1.23);
}
