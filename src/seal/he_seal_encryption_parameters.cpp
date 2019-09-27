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

#include "seal/he_seal_encryption_parameters.hpp"
#include "logging/ngraph_he_log.hpp"
#include "ngraph/check.hpp"
#include "ngraph/except.hpp"
#include "seal/seal_util.hpp"

ngraph::he::HESealEncryptionParameters::HESealEncryptionParameters(
    const std::string& scheme_name, const seal::EncryptionParameters& parms,
    std::uint64_t security_level, double scale, bool complex_packing)
    : m_scheme_name(scheme_name),
      m_seal_encryption_parameters(parms),
      m_security_level(security_level),
      m_scale(scale),
      m_complex_packing(complex_packing) {
  validate_parameters();
}

ngraph::he::HESealEncryptionParameters::HESealEncryptionParameters()
    : HESealEncryptionParameters(
          "HE_SEAL", 1024, std::vector<int>{30, 30, 30, 30, 30}, 0, 0, false) {}

ngraph::he::HESealEncryptionParameters::HESealEncryptionParameters(
    const std::string& scheme_name, std::uint64_t poly_modulus_degree,
    std::vector<int> coeff_modulus_bits, std::uint64_t security_level,
    double scale, bool complex_packing)
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

void ngraph::he::HESealEncryptionParameters::validate_parameters() const {
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

  auto seal_sec_level = ngraph::he::seal_security_level(security_level());

  auto context = seal::SEALContext::Create(m_seal_encryption_parameters, true,
                                           seal_sec_level);

  NGRAPH_CHECK(context->parameters_set(), "Invalid parameters");

  // TODO: validate scale is reasonable
}

void ngraph::he::HESealEncryptionParameters::save(std::ostream& stream) const {
  NGRAPH_HE_LOG(5) << "Saving scale " << m_scale;
  stream.write(reinterpret_cast<const char*>(&m_scale), sizeof(m_scale));
  NGRAPH_HE_LOG(5) << "Saving complex packing " << m_complex_packing;
  stream.write(reinterpret_cast<const char*>(&m_complex_packing),
               sizeof(m_complex_packing));

  NGRAPH_HE_LOG(5) << "Saving security level " << m_security_level;
  stream.write(reinterpret_cast<const char*>(&m_security_level),
               sizeof(m_security_level));

  NGRAPH_HE_LOG(5) << "Saving parms";
  seal::EncryptionParameters::Save(m_seal_encryption_parameters, stream);
}

ngraph::he::HESealEncryptionParameters
ngraph::he::HESealEncryptionParameters::load(std::istream& stream) {
  NGRAPH_HE_LOG(5) << "Loading scale";
  double scale;
  stream.read(reinterpret_cast<char*>(&scale), sizeof(scale));
  NGRAPH_HE_LOG(5) << "Loaded scale " << scale;

  bool complex_packing;
  stream.read(reinterpret_cast<char*>(&complex_packing),
              sizeof(complex_packing));
  NGRAPH_HE_LOG(5) << "Loaded complex_packing " << complex_packing;

  uint64_t security_level;
  stream.read(reinterpret_cast<char*>(&security_level), sizeof(security_level));
  NGRAPH_HE_LOG(5) << "Loaded security_level " << security_level;
  auto seal_encryption_parameters = seal::EncryptionParameters::Load(stream);

  return HESealEncryptionParameters("HE_SEAL", seal_encryption_parameters,
                                    security_level, scale, complex_packing);
}

ngraph::he::HESealEncryptionParameters
ngraph::he::HESealEncryptionParameters::parse_config_or_use_default(
    const char* config_path) {
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

    NGRAPH_CHECK(parsed_scheme_name == "HE_SEAL", "Parsed scheme name ",
                 parsed_scheme_name, " is not HE_SEAL");

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
    throw ngraph::ngraph_error(ss.str());
  }
}