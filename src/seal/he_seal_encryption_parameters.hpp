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
/// \brief Class representing CKKS encryption parameters
class HESealEncryptionParameters {
 public:
  HESealEncryptionParameters() = delete;

  /// \brief Constructs CKKS encryption parameters
  /// \param[in] scheme_name Should be "HE_SEAL"
  /// \param[in] poly_modulus_degree Degree of the RLWE polynomial. Should be a
  /// power of 2 \param[in] security_level Bits of security. 0 indicates no
  /// security
  /// \param[in] scale Scale at which to encode. Roughly corresponds to the
  /// precision of the computation
  ///  \param[in] coeff_modulus_bits Vector of bit-widths of the cofficient
  ///  moduli.
  /// \throws ngraph_error if scheme_name is not "HE_SEAL"
  HESealEncryptionParameters(const std::string& scheme_name,
                             std::uint64_t poly_modulus_degree,
                             std::uint64_t security_level, double scale,
                             std::vector<int> coeff_modulus_bits)
      : m_scheme_name(scheme_name),
        m_poly_modulus_degree(poly_modulus_degree),
        m_security_level(security_level),
        m_scale(scale),
        m_coeff_modulus_bits(coeff_modulus_bits) {
    NGRAPH_CHECK(scheme_name == "HE_SEAL", "Invalid scheme name ", scheme_name);
    m_seal_encryption_parameters =
        seal::EncryptionParameters(seal::scheme_type::CKKS);

    m_seal_encryption_parameters.set_poly_modulus_degree(poly_modulus_degree);

    m_coeff_modulus =
        seal::CoeffModulus::Create(poly_modulus_degree, coeff_modulus_bits);

    m_seal_encryption_parameters.set_coeff_modulus(m_coeff_modulus);
  }

  /// \brief Saves encryption parameters to a stream
  void save(std::ostream& stream) const {
    stream << m_scale;
    seal::EncryptionParameters::Save(m_seal_encryption_parameters, stream);
  }

  /// \brief Returns SEAL encryption parameters
  seal::EncryptionParameters& seal_encryption_parameters() {
    return m_seal_encryption_parameters;
  }

  /// \brief Returns SEAL encryption parameters
  const seal::EncryptionParameters& seal_encryption_parameters() const {
    return m_seal_encryption_parameters;
  }

  /// \brief Returns the scheme name
  inline const std::string& scheme_name() const { return m_scheme_name; }

  /// \brief Returns the polynomial modulus degree
  inline std::uint64_t poly_modulus_degree() const {
    return m_poly_modulus_degree;
  }

  /// \brief Returns the scale
  inline double scale() const { return m_scale; }

  /// \brief Returns the security level
  inline std::uint64_t security_level() const { return m_security_level; }

  /// \brief Returns the vector of bit-widths of the coefficient moduli
  inline const std::vector<int>& coeff_modulus_bits() const {
    return m_coeff_modulus_bits;
  }

  /// \brief Returns the vector of coefficient moduli
  inline const std::vector<seal::SmallModulus>& coeff_modulus() const {
    return m_coeff_modulus;
  }

 private:
  seal::EncryptionParameters m_seal_encryption_parameters{
      seal::scheme_type::CKKS};
  std::string m_scheme_name;
  std::uint64_t m_poly_modulus_degree;
  std::uint64_t m_security_level;
  double m_scale;
  std::vector<int> m_coeff_modulus_bits;
  std::vector<seal::SmallModulus> m_coeff_modulus;
};

/// \brief Returns a set of default CKKS parameters
/// \warning Default parameters do not enforce any security level
inline ngraph::he::HESealEncryptionParameters default_ckks_parameters() {
  size_t poly_modulus_degree = 1024;
  size_t security_level = 0;  // No enforced security level
  std::vector<int> coeff_modulus_bits = {30, 30, 30, 30, 30};
  double default_scale = 0;  // Use default scale

  auto params = ngraph::he::HESealEncryptionParameters(
      "HE_SEAL", poly_modulus_degree, security_level, default_scale,
      coeff_modulus_bits);
  return params;
}

/// \brief Returns encryption parameters at given path if possible, or use
/// default parameters
/// \param[in] config_path filename where configuration is
/// stored. If empty, uses default configuration
/// \throws ngraph_error if config_path is specified but does not exist
/// \throws ngraph_error if encryption parameters are not valid
inline ngraph::he::HESealEncryptionParameters parse_config_or_use_default(
    const char* config_path) {
  if (config_path == nullptr) {
    return default_ckks_parameters();
  }

  auto file_exists = [](const char* filename) {
    std::ifstream f(filename);
    return f.good();
  };
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
    if (parsed_scheme_name != "HE_SEAL") {
      throw ngraph_error("Parsed scheme name " + parsed_scheme_name +
                         " is not HE_SEAL");
    }

    uint64_t poly_modulus_degree = js["poly_modulus_degree"];
    uint64_t security_level = js["security_level"];

    static std::unordered_set<uint64_t> valid_poly_modulus{1024, 2048,  4096,
                                                           8192, 16384, 32768};
    if (valid_poly_modulus.count(poly_modulus_degree) == 0) {
      throw ngraph_error(
          "poly_modulus_degree must be 1024, 2048, 4096, 8192, 16384, "
          "32768");
    }

    static std::unordered_set<uint64_t> valid_security_level{0, 128, 192, 256};
    if (valid_security_level.count(security_level) == 0) {
      throw ngraph_error("security_level must be 0, 128, 192, 256");
    }

    double scale = 0;  // Use default scale
    if (js.find("scale") != js.end()) {
      scale = js["scale"];
    }

    std::vector<int> coeff_mod_bits = js["coeff_modulus"];
    for (const auto& coeff_bit : coeff_mod_bits) {
      if (coeff_bit > 60 || coeff_bit < 1) {
        NGRAPH_ERR << "coeff_bit " << coeff_bit;
        throw ngraph_error("Invalid coeff modulus");
      }
    }

    auto params = ngraph::he::HESealEncryptionParameters(
        "HE_SEAL", poly_modulus_degree, security_level, scale, coeff_mod_bits);

    return params;

  } catch (const std::exception& e) {
    std::stringstream ss;
    ss << "Error creating encryption parameters: " << e.what();
    throw ngraph_error(ss.str());
  }
}
}  // namespace he
}  // namespace ngraph
