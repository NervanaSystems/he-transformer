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

#pragma once

#include <cstdlib>
#include <fstream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <unordered_set>

#include "nlohmann/json.hpp"
#include "seal/seal.h"

namespace ngraph {
namespace he {
class HESealEncryptionParameters {
 public:
  HESealEncryptionParameters() = delete;
  HESealEncryptionParameters(const std::string& scheme_name,
                             std::uint64_t poly_modulus_degree,
                             std::uint64_t security_level,
                             std::vector<int> coeff_modulus_bits)
      : m_scheme_name(scheme_name),
        m_poly_modulus_degree(poly_modulus_degree),
        m_security_level(security_level),
        m_coeff_modulus_bits(coeff_modulus_bits) {
    NGRAPH_CHECK(scheme_name == "HE_SEAL", "Invalid scheme name ", scheme_name);
    m_seal_encryption_parameters =
        seal::EncryptionParameters(seal::scheme_type::CKKS);

    m_seal_encryption_parameters.set_poly_modulus_degree(poly_modulus_degree);

    m_coeff_modulus =
        seal::CoeffModulus::Create(poly_modulus_degree, coeff_modulus_bits);

    m_seal_encryption_parameters.set_coeff_modulus(m_coeff_modulus);
  }

  void save(std::ostream& stream) const {
    seal::EncryptionParameters::Save(m_seal_encryption_parameters, stream);
  }

  seal::EncryptionParameters& seal_encryption_parameters() {
    return m_seal_encryption_parameters;
  }
  const seal::EncryptionParameters& seal_encryption_parameters() const {
    return m_seal_encryption_parameters;
  }

  inline const std::string& scheme_name() const { return m_scheme_name; }

  inline std::uint64_t poly_modulus_degree() const {
    return m_poly_modulus_degree;
  }

  inline std::uint64_t security_level() const { return m_security_level; }

  inline const std::vector<int>& coeff_modulus_bits() const {
    return m_coeff_modulus_bits;
  }

  inline const std::vector<seal::SmallModulus>& coeff_modulus() const {
    return m_coeff_modulus;
  }

 private:
  seal::EncryptionParameters m_seal_encryption_parameters{
      seal::scheme_type::CKKS};
  std::string m_scheme_name;
  std::uint64_t m_poly_modulus_degree;
  std::uint64_t m_security_level;
  std::vector<int> m_coeff_modulus_bits;
  std::vector<seal::SmallModulus> m_coeff_modulus;
};

inline ngraph::he::HESealEncryptionParameters default_ckks_parameters() {
  size_t poly_modulus_degree = 1024;
  size_t security_level = 0;  // No enforced security level
  std::vector<int> coeff_modulus_bits = {30, 30, 30, 30, 30};

  auto params = ngraph::he::HESealEncryptionParameters(
      "HE_SEAL", poly_modulus_degree, security_level, coeff_modulus_bits);
  return params;
}

inline ngraph::he::HESealEncryptionParameters parse_config_or_use_default(
    const std::string& scheme_name) {
  std::unordered_set<std::string> valid_scheme_names{"HE_SEAL"};
  if (valid_scheme_names.find(scheme_name) == valid_scheme_names.end()) {
    throw ngraph_error("Invalid scheme name " + scheme_name);
  }

  const char* config_path = getenv("NGRAPH_HE_SEAL_CONFIG");
  if (config_path == nullptr) {
    return default_ckks_parameters();
  }

  auto file_exists = [](const char* filename) {
    std::ifstream f(filename);
    return f.good();
  };
  if (!file_exists(config_path)) {
    NGRAPH_INFO << "Config path" << config_path << " does not exist";
  }
  NGRAPH_CHECK(file_exists(config_path), "Config path ", config_path,
               " does not exist");

  try {
    // Read file to string
    std::ifstream f(config_path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();

    // Parse json
    nlohmann::json js = nlohmann::json::parse(s);
    std::string parsed_scheme_name = js["scheme_name"];
    if (parsed_scheme_name != scheme_name) {
      NGRAPH_INFO << "Parsed scheme name " << parsed_scheme_name
                  << " doesn't match scheme name " << scheme_name;
      throw ngraph_error("Parsed scheme name " + parsed_scheme_name +
                         " doesn't match scheme name " + scheme_name);
    }

    uint64_t poly_modulus_degree = js["poly_modulus_degree"];
    uint64_t security_level = js["security_level"];
    std::unordered_set<uint64_t> valid_poly_modulus{1024, 2048,  4096,
                                                    8192, 16384, 32768};
    if (valid_poly_modulus.count(poly_modulus_degree) == 0) {
      throw ngraph_error(
          "poly_modulus_degree must be 1024, 2048, 4096, 8192, 16384, "
          "32768");
    }

    std::unordered_set<uint64_t> valid_security_level{0, 128, 192, 256};
    if (valid_security_level.count(security_level) == 0) {
      throw ngraph_error("security_level must be 0, 128, 192, 256");
    }

    std::vector<int> coeff_mod_bits = js["coeff_modulus"];

    for (const auto& coeff_bit : coeff_mod_bits) {
      if (coeff_bit > 60 || coeff_bit < 1) {
        NGRAPH_INFO << "coeff_bit " << coeff_bit;
        throw ngraph_error("Invalid coeff modulus");
      }
    }
    NGRAPH_INFO << "Creating parms";
    auto params = ngraph::he::HESealEncryptionParameters(
        scheme_name, poly_modulus_degree, security_level, coeff_mod_bits);
    NGRAPH_INFO << "returning parms";

    return params;

  } catch (const std::exception& e) {
    std::stringstream ss;
    ss << "Error parsing NGRAPH_HE_SEAL_CONFIG: " << e.what();
    NGRAPH_ERR << ss.str();
    throw ngraph_error(ss.str());
  }
}
}  // namespace he
}  // namespace ngraph
