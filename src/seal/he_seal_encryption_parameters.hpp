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
  /// \brief Constructs encryption parameteters from SEAL parameters
  /// \param[in] scheme_name Should be "HE_SEAL"
  /// \param[in] parms SEAL encryption parameters
  ///  \param[in] security_level Bits of security.0 indicates no security
  /// \param[in] scale Scale at which to encode. Roughly corresponds to the
  /// precision of the computation
  /// \param[in] complex_packing Whether or not to pack scalars (a,b,c,d) as (a
  /// +bi, c+di)
  HESealEncryptionParameters(const std::string& scheme_name,
                             const seal::EncryptionParameters& parms,
                             std::uint64_t security_level, double scale,
                             bool complex_packing)
      : m_scheme_name(scheme_name),
        m_seal_encryption_parameters(parms),
        m_security_level(security_level),
        m_scale(scale),
        m_complex_packing(complex_packing) {
    validate_parameters();
  }

  /// \brief Returns a set of default CKKS parameters
  /// \warning Default parameters do not enforce any security level
  HESealEncryptionParameters()
      : HESealEncryptionParameters("HE_SEAL", 1024,
                                   std::vector<int>{30, 30, 30, 30, 30}, 0, 0,
                                   false) {}

  /// \brief Constructs CKKS encryption parameters
  /// \param[in] scheme_name Should be "HE_SEAL"
  /// \param[in] poly_modulus_degree Degree of the RLWE polynomial. Should be a
  /// power of 2
  /// \param[in] security_level Bits of security. 0 indicates no security
  /// \param[in] scale Scale at which to encode. Roughly corresponds to the
  /// precision of the computation
  /// \param[in] complex_packing Whether or not to pack scalars (a,b,c,d) as (a
  /// +bi, c+di)
  ///  \param[in] coeff_modulus_bits Vector of bit-widths of the cofficient
  ///  moduli.
  /// \throws ngraph_error if scheme_name is not "HE_SEAL"
  HESealEncryptionParameters(const std::string& scheme_name,
                             std::uint64_t poly_modulus_degree,
                             std::vector<int> coeff_modulus_bits,
                             std::uint64_t security_level, double scale,
                             bool complex_packing)
      : m_scheme_name(scheme_name),
        m_security_level(security_level),
        m_scale(scale),
        m_complex_packing(complex_packing) {
    m_seal_encryption_parameters =
        seal::EncryptionParameters(seal::scheme_type::CKKS);

    m_seal_encryption_parameters.set_poly_modulus_degree(poly_modulus_degree);

    auto coeff_modulus =
        seal::CoeffModulus::Create(poly_modulus_degree, coeff_modulus_bits);

    m_seal_encryption_parameters.set_coeff_modulus(coeff_modulus);

    validate_parameters();
  }

  /// \brief Checks paramters are valid
  /// \throws ngraph_error if scheme_name is not HE_SEAL
  /// \throws ngraph_error if poly_modulus_degree is not a supported power of 2
  /// \throws ngraph_error if security level is not valid security
  /// level
  inline void validate_parameters() {
    NGRAPH_CHECK(m_scheme_name == "HE_SEAL", "Invalid scheme name ",
                 m_scheme_name);

    static std::unordered_set<uint64_t> valid_poly_modulus{1024, 2048,  4096,
                                                           8192, 16384, 32768};
    NGRAPH_CHECK(valid_poly_modulus.count(poly_modulus_degree()) != 0,
                 "poly_modulus_degree must be 1024, 2048, 4096, 8192, 16384, "
                 "32768");

    static std::unordered_set<uint64_t> valid_security_level{0, 128, 192, 256};

    NGRAPH_CHECK(valid_security_level.count(security_level()) != 0,
                 "security_level must be 0, 128, 192, 256");
  }

  /// \brief Saves encryption parameters to a stream
  inline void save(std::ostream& stream) const {
    stream << m_scale;
    stream << m_complex_packing;
    stream << m_security_level;
    seal::EncryptionParameters::Save(m_seal_encryption_parameters, stream);
  }

  static inline HESealEncryptionParameters load(std::istream& stream) {
    double scale;
    stream >> scale;
    bool complex_packing;
    stream >> complex_packing;
    uint64_t security_level;
    stream >> security_level;
    auto seal_encryption_parameters = seal::EncryptionParameters::Load(stream);

    return HESealEncryptionParameters("HE_SEAL", seal_encryption_parameters,
                                      security_level, scale, complex_packing);
  }

  /// \brief Returns SEAL encryption parameters
  inline seal::EncryptionParameters& seal_encryption_parameters() {
    return m_seal_encryption_parameters;
  }

  /// \brief Returns SEAL encryption parameters
  inline const seal::EncryptionParameters& seal_encryption_parameters() const {
    return m_seal_encryption_parameters;
  }

  /// \brief Returns the scheme name
  inline const std::string& scheme_name() const { return m_scheme_name; }

  /// \brief Returns the polynomial modulus degree
  inline std::uint64_t poly_modulus_degree() const {
    return m_seal_encryption_parameters.poly_modulus_degree();
  }

  /// \brief Returns the scale
  inline double scale() const { return m_scale; }

  /// \brief Returns the security level
  inline std::uint64_t security_level() const { return m_security_level; }

  /// \brief Return whether or not complex packing is enabled
  inline bool complex_packing() const { return m_complex_packing; }

  /// \brief Return whether or not complex packing is enabled
  inline bool& complex_packing() { return m_complex_packing; }

 private:
  std::string m_scheme_name;
  seal::EncryptionParameters m_seal_encryption_parameters{
      seal::scheme_type::CKKS};
  std::uint64_t m_security_level;
  double m_scale;
  bool m_complex_packing;
};

/// \brief Returns encryption parameters at given path if possible, or use
/// default parameters
/// \param[in] config_path filename where configuration is
/// stored. If empty, uses default configuration
/// \throws ngraph_error if config_path is specified but does not exist
/// \throws ngraph_error if encryption parameters are not valid
static inline ngraph::he::HESealEncryptionParameters
parse_config_or_use_default(const char* config_path) {
  if (config_path == nullptr) {
    return ngraph::he::HESealEncryptionParameters();
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

    double scale = 0;  // Use default scale
    if (js.find("scale") != js.end()) {
      scale = js["scale"];
    }

    std::vector<int> coeff_mod_bits = js["coeff_modulus"];

    bool complex_packing = false;
    if (js.find("complex_packing") != js.end()) {
      complex_packing = js["complex_packing"];
    }

    auto params = ngraph::he::HESealEncryptionParameters(
        "HE_SEAL", poly_modulus_degree, coeff_mod_bits, security_level, scale,
        complex_packing);

    return params;

  } catch (const std::exception& e) {
    std::stringstream ss;
    ss << "Error creating encryption parameters: " << e.what();
    throw ngraph_error(ss.str());
  }
}
}  // namespace he
}  // namespace ngraph
