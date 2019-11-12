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

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "gtest/gtest.h"
#include "logging/ngraph_he_log.hpp"
#include "ngraph/file_util.hpp"
#include "seal/he_seal_encryption_parameters.hpp"
#include "seal/seal.h"
#include "seal/seal_util.hpp"

TEST(encryption_parameters, create) {
  size_t poly_modulus_degree{4096};
  auto seal_encryption_parameters =
      seal::EncryptionParameters(seal::scheme_type::CKKS);

  seal_encryption_parameters.set_poly_modulus_degree(poly_modulus_degree);

  auto coeff_modulus = seal::CoeffModulus::Create(poly_modulus_degree,
                                                  std::vector<int>{30, 30, 30});

  seal_encryption_parameters.set_coeff_modulus(coeff_modulus);

  // Baseline parameters
  EXPECT_NO_THROW(ngraph::he::HESealEncryptionParameters(
      "HE_SEAL", seal_encryption_parameters, 128, 1.23, false));

  // Incorrect scheme name
  EXPECT_ANY_THROW(ngraph::he::HESealEncryptionParameters(
      "DUMMY_NAME", seal_encryption_parameters, 128, 1.23, false));

  // Incorrect security level
  EXPECT_ANY_THROW(ngraph::he::HESealEncryptionParameters(
      "HE_SEAL", seal_encryption_parameters, 123, 1.23, false));

  // Parameters violate security level
  EXPECT_ANY_THROW(ngraph::he::HESealEncryptionParameters(
      "HE_SEAL", seal_encryption_parameters, 192, 1.23, false));

  EXPECT_ANY_THROW(ngraph::he::HESealEncryptionParameters(
      "HE_SEAL", seal_encryption_parameters, 256, 1.23, false));

  // No enforced security level
  EXPECT_NO_THROW(ngraph::he::HESealEncryptionParameters(
      "HE_SEAL", seal_encryption_parameters, 0, 1.23, false));

  // Attributes set properly
  auto parms = ngraph::he::HESealEncryptionParameters(
      "HE_SEAL", seal_encryption_parameters, 128, 1.23, true);
  EXPECT_EQ(parms.complex_packing(), true);
  EXPECT_EQ(parms.scale(), 1.23);
  EXPECT_EQ(parms.security_level(), 128);
}

TEST(encryption_parameters, save) {
  size_t poly_modulus_degree{4096};
  auto seal_encryption_parameters =
      seal::EncryptionParameters(seal::scheme_type::CKKS);

  seal_encryption_parameters.set_poly_modulus_degree(poly_modulus_degree);

  auto coeff_modulus = seal::CoeffModulus::Create(poly_modulus_degree,
                                                  std::vector<int>{30, 30, 30});

  seal_encryption_parameters.set_coeff_modulus(coeff_modulus);

  auto he_parms = ngraph::he::HESealEncryptionParameters(
      "HE_SEAL", seal_encryption_parameters, 128, 1.23, false);

  std::stringstream ss;
  he_parms.save(ss);

  auto loaded_parms = ngraph::he::HESealEncryptionParameters::load(ss);

  EXPECT_EQ(he_parms.scale(), loaded_parms.scale());
  EXPECT_EQ(he_parms.complex_packing(), loaded_parms.complex_packing());
  EXPECT_EQ(he_parms.seal_encryption_parameters(),
            loaded_parms.seal_encryption_parameters());
}

TEST(encryption_parameters, from_string) {
  std::string param_str = R"(
    {
        "scheme_name" : "HE_SEAL",
        "poly_modulus_degree" : 2048,
        "security_level" : 128,
        "coeff_modulus" : [54],
        "scale" : 1.23,
        "complex_packing" : true
    })";
  auto he_parms =
      ngraph::he::HESealEncryptionParameters::parse_config_or_use_default(
          param_str.c_str());

  EXPECT_EQ(he_parms.poly_modulus_degree(), 2048);
  EXPECT_EQ(he_parms.security_level(), 128);
  EXPECT_EQ(he_parms.scale(), 1.23);
  EXPECT_EQ(he_parms.complex_packing(), true);
}

TEST(encryption_parameters, from_string_invalid) {
  std::string param_str = R"(
    {
        "DUMMY_STR"
    })";

  EXPECT_ANY_THROW(
      ngraph::he::HESealEncryptionParameters::parse_config_or_use_default(
          param_str.c_str()));
}

TEST(encryption_parameters, from_file) {
  std::string config = R"(
    {
        "scheme_name": "HE_SEAL",
        "poly_modulus_degree": 2048,
        "security_level": 128,
        "coeff_modulus": [54],
        "scale": 1.23,
        "complex_packing": true
    }
    )";

  std::string filename = ngraph::file_util::tmp_filename();
  {
    std::ofstream file(filename);
    file << config;
  }

  auto he_parms =
      ngraph::he::HESealEncryptionParameters::parse_config_or_use_default(
          filename.c_str());

  EXPECT_EQ(he_parms.poly_modulus_degree(), 2048);
  EXPECT_EQ(he_parms.security_level(), 128);

  EXPECT_EQ(he_parms.scale(), 1.23);
  EXPECT_EQ(he_parms.complex_packing(), true);

  ngraph::file_util::remove_file(filename);
}

TEST(encryption_parameters, choose_scale) {
  auto test_choose_scale = [](const std::vector<int>& coeff_modulus_bits) {
    std::string param_str = R"(
    {
        "scheme_name" : "HE_SEAL",
        "poly_modulus_degree" : 2048,
        "security_level" : 0,
        "coeff_modulus" : XXXXXXX,
        "complex_packing" : true
    })";
    std::string filler_str{"XXXXXXX"};

    std::string coeff_mod_str = ngraph::vector_to_string(coeff_modulus_bits);
    param_str.replace(param_str.find(filler_str), filler_str.length(),
                      coeff_mod_str);

    auto he_parms =
        ngraph::he::HESealEncryptionParameters::parse_config_or_use_default(
            param_str.c_str());

    double exp_scale = ngraph::he::HESealEncryptionParameters::choose_scale(
        seal::CoeffModulus::Create(he_parms.poly_modulus_degree(),
                                   coeff_modulus_bits));

    EXPECT_EQ(he_parms.poly_modulus_degree(), 2048);
    EXPECT_EQ(he_parms.security_level(), 0);
    EXPECT_EQ(he_parms.scale(), exp_scale);
    EXPECT_EQ(he_parms.complex_packing(), true);
  };

  test_choose_scale(std::vector<int>{54});
  test_choose_scale(std::vector<int>{54, 54});
  test_choose_scale(std::vector<int>{54, 54, 54});
}
