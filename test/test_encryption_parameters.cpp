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

#include "gtest/gtest.h"
#include "seal/he_seal_encryption_parameters.hpp"
#include "seal/seal.h"

using namespace std;
using namespace ngraph::he;

TEST(encryption_parameters, create) {
  size_t poly_modulus_degree{4096};

  auto seal_encryption_parameters =
      seal::EncryptionParameters(seal::scheme_type::CKKS);

  seal_encryption_parameters.set_poly_modulus_degree(poly_modulus_degree);

  auto coeff_modulus = seal::CoeffModulus::Create(poly_modulus_degree,
                                                  std::vector<int>{30, 30, 30});

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

}

