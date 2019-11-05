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

#include <cstdint>
#include <iostream>
#include <string>

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
  HESealEncryptionParameters(std::string scheme_name,
                             seal::EncryptionParameters parms,
                             std::uint64_t security_level, double scale,
                             bool complex_packing);

  /// \brief Returns a set of default CKKS parameters using real packing
  /// \warning Default parameters do not enforce any security level
  static HESealEncryptionParameters default_real_packing_parms();

  /// \brief Returns a set of default CKKS parameters using complex packing
  /// \warning Default parameters do not enforce any security level
  static HESealEncryptionParameters default_complex_packing_parms();

  /// \brief Returns a set of default CKKS parameters
  /// \warning Default parameters do not enforce any security level
  HESealEncryptionParameters();

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
  HESealEncryptionParameters(std::string scheme_name,
                             std::uint64_t poly_modulus_degree,
                             std::vector<int> coeff_modulus_bits,
                             std::uint64_t security_level, double scale,
                             bool complex_packing);

  /// \brief Returns encryption parameters at given path if possible, or use
  /// default parameters
  /// \param[in] config filename where configuration is stored, or contents of
  /// filename with configuration. If empty, uses default configuration
  /// \throws ngraph_error if config_path is specified but does not exist
  /// \throws ngraph_error if encryption parameters are not valid
  static HESealEncryptionParameters parse_config_or_use_default(
      const char* config);

  /// \brief Returns whether or not all fields match
  /// \param[in] other Encryption parameters to compare against
  bool operator==(const HESealEncryptionParameters& other) const;

  /// \brief Returns whether or not two encryption parameters differ in such a
  /// way they can use the same SEAL context
  /// \param[in] parms1 Encryption parameter
  /// \param[in] parms2 Encryption parameter
  static bool same_context(const HESealEncryptionParameters& parms1,
                           const HESealEncryptionParameters& parms2);

  /// \brief Checks paramters are valid
  /// \throws ngraph_error if scheme_name is not HE_SEAL
  /// \throws ngraph_error if poly_modulus_degree is not a supported power of 2
  /// \throws ngraph_error if security level is not valid security
  /// level
  void validate_parameters() const;

  /// \brief Saves encryption parameters to a stream
  void save(std::ostream& stream) const;

  /// \brief Loads encryption parametrs from a stream
  static HESealEncryptionParameters load(std::istream& stream);

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

  /// \brief Returns the scale
  /// TODO(fboemer): verify scale is valid
  inline double& scale() { return m_scale; }

  /// \brief Returns the security level
  inline std::uint64_t security_level() const { return m_security_level; }

  /// \brief Returns the security level
  inline std::uint64_t& security_level() { return m_security_level; }

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

/// \brief Prints the given encryption parameters
/// \param[in] params Encryption parameters
/// \param[in] context SEAL context associated with parameters
void print_encryption_parameters(const HESealEncryptionParameters& params,
                                 const seal::SEALContext& context);

}  // namespace he
}  // namespace ngraph
