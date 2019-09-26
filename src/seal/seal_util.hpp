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

#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include "ngraph/check.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"

namespace ngraph {
namespace he {
class SealCiphertextWrapper;
class SealPlaintextWrapper;
class HESealBackend;

/// \brief Prints the given SEAL context
/// \param[in] context SEAL context to print
void print_seal_context(const seal::SEALContext& context);

/// \brief Chooses a default scale for the given list of coefficient moduli
/// \param[in] coeff_moduli List of coefficient moduli
/// \returns Scale
inline double choose_scale(
    const std::vector<seal::SmallModulus>& coeff_moduli) {
  if (coeff_moduli.size() > 2) {
    return static_cast<double>(coeff_moduli[coeff_moduli.size() - 2].value());
  } else if (coeff_moduli.size() > 1) {
    return static_cast<double>(coeff_moduli.back().value()) / 4096.0;
  } else {
    // Enable a single multiply
    return sqrt(static_cast<double>(coeff_moduli.back().value() / 256.0));
  }
}

/// \brief Returns the smallest chain index of a vector of ciphertexts
/// \param[in] ciphers Vector of ciphertexts
/// \param[in] he_seal_backend Backend whose context is used to determine the
/// chain index
/// \returns The minimum chain index of the ciphertexts in ciphers
/// TODO: move to he_seal_backend
size_t match_to_smallest_chain_index(
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& ciphers,
    const HESealBackend& he_seal_backend);

/// \brief Returns whether or not two cipher/plaintexts have a similar scale
/// \param[in] arg0 Ciphertext or plaintext
/// \param[in] arg1 Ciphertext or plaintext
/// \param[in] factor Multiplicative tolerance within which two
/// cipher/plaintexts are considered to have the same scale
template <typename S, typename T>
inline bool within_rescale_tolerance(const S& arg0, const T& arg1,
                                     double factor = 1.05) {
  const auto scale0 = arg0.scale();
  const auto scale1 = arg1.scale();

  bool within_tolerance =
      (scale0 / scale1 <= factor && scale1 / scale0 <= factor);
  return within_tolerance;
}

/// \brief Matches the scale of two cipher/plaintexts with similar scale
/// \param[in,out] arg0 Ciphertext or plaintext whose scale will be adjusted
/// \param[in] arg1 Ciphertext or plaintext whose scale is changed to
/// \throws ngraph_error if cipher/plaintexts do not have similar scales
template <typename S, typename T>
inline void match_scale(S& arg0, const T& arg1) {
  auto scale0 = arg0.scale();
  auto scale1 = arg1.scale();
  bool scale_ok = within_rescale_tolerance(arg0, arg1);
  NGRAPH_CHECK(scale_ok, "Scale ", scale0, " does not match scale ", scale1);
  arg0.scale() = arg1.scale();
}

/// \brief Matches the scale and level of two ciphertexts with similar scale.
/// Uses modulus switching and scaling as deduced by the similarity of the
/// scales
/// \param[in,out] arg0 Ciphertext
/// \param[in,out] arg1 Ciphertext
/// \param[in] he_seal_backend Backend whose context is used for rescaling
/// \param[in] pool Memory pool used for rescaling
void match_modulus_and_scale_inplace(
    SealCiphertextWrapper& arg0, SealCiphertextWrapper& arg1,
    const HESealBackend& he_seal_backend,
    seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());

/// \brief Adds a ciphertext with a scalar in every slot
/// \param[in,out] encrypted Ciphertext to add to.
/// \param[in] value Value which is added to the ciphertext
/// \param[in] he_seal_backend Backend whose context is used for encoding and
/// addition
void add_plain_inplace(seal::Ciphertext& encrypted, double value,
                       const HESealBackend& he_seal_backend);

/// \brief Adds a ciphertext with a scalar in every slot
/// \param[in] encrypted Ciphertext to add to.
/// \param[in] value Value which is added to the ciphertext
/// \param[out] destination Ciphertext storing the result
/// \param[in] he_seal_backend Backend whose context is used for encoding and
/// addition
inline void add_plain(const seal::Ciphertext& encrypted, double value,
                      seal::Ciphertext& destination,
                      const HESealBackend& he_seal_backend) {
  destination = encrypted;
  ngraph::he::add_plain_inplace(destination, value, he_seal_backend);
}

/// \brief Multiples each element in a polynomial with a scalar modulo
/// modulus_value. Assumes the scalar, poly, and modulus value are all < 30 bits
/// \param[in] poly Polynomial to be multiplied
/// \param[in] coeff_count Number of terms in the polynomial
/// \param[in] scalar Value with which to multiply
/// \param[in] modulus_value modulus with which to reduce each product
/// \param[in] const_ratio TODO
/// \param[out] result Will store the result of the multiplication
void multiply_poly_scalar_coeffmod64(const uint64_t* poly, size_t coeff_count,
                                     uint64_t scalar,
                                     const std::uint64_t modulus_value,
                                     const std::uint64_t const_ratio,
                                     uint64_t* result);

/// \brief Adds each element in a polynomial with a scalar modulo
/// modulus_value.
/// \param[in] poly Polynomial to be multiplied
/// \param[in] coeff_count Number of terms in the polynomial
/// \param[in] scalar Value with which to add
/// \param[in] modulus modulus with which to reduce each addition
/// \param[out] result Will store the result of the multiplication
inline void add_poly_scalar_coeffmod(const std::uint64_t* poly,
                                     std::size_t coeff_count,
                                     std::uint64_t scalar,
                                     const seal::SmallModulus& modulus,
                                     std::uint64_t* result) {
  const uint64_t modulus_value = modulus.value();
#ifdef SEAL_DEBUG
  if (poly == nullptr && coeff_count > 0) {
    throw ngraph_error("poly");
  }
  if (scalar >= modulus_value) {
    throw ngraph_error("scalar");
  }
  if (modulus.is_zero()) {
    throw ngraph_error("modulus");
  }
  if (result == nullptr && coeff_count > 0) {
    throw ngraph_error("result");
  }
#endif

  for (; coeff_count--; result++, poly++) {
// Explicit inline
// result[i] = add_uint_uint_mod(poly[i], scalar, modulus);
#ifdef SEAL_DEBUG
    if (*poly >= modulus_value) {
      throw ngraph_error("poly > modulus_value");
    }

#endif
    std::uint64_t sum = *poly + scalar;
    *result = sum - (modulus_value &
                     static_cast<std::uint64_t>(
                         -static_cast<std::int64_t>(sum >= modulus_value)));
  }
}

/// \brief Multiplies a ciphertext with a scalar in every slot
/// \param[in,out] encrypted Ciphertext to multply
/// \param[in] value Multiplicand multiplied with the ciphertext
/// \param[in] he_seal_backend Backend whose context is used for encoding and
/// multiplication
/// \param[in] pool Memory pool used for new memory allocation
void multiply_plain_inplace(
    seal::Ciphertext& encrypted, double value,
    const HESealBackend& he_seal_backend,
    seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());

/// \brief Multiplies a ciphertext with a scalar in every slot
/// \param[in] encrypted Ciphertext to multply
/// \param[in] value Multiplicand multiplied with the ciphertext
/// \param[out] destination Ciphertext storing the result
/// \param[in] he_seal_backend Backend whose context is used for encoding and
/// multiplication
/// \param[in] pool Memory pool used for new memory allocation
inline void multiply_plain(
    const seal::Ciphertext& encrypted, double value,
    seal::Ciphertext& destination, const HESealBackend& he_seal_backend,
    seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {
  destination = encrypted;
  ngraph::he::multiply_plain_inplace(destination, value, he_seal_backend,
                                     std::move(pool));
}

/// \brief Optimized encoding of single value into vector of coefficients
/// \param[in] value Value to be encoded
/// \param[in] element_type TODO: remove
/// \param[in] scale Scale at which to encode value
/// \param[in] parms_id Seal parameter id to use in encoding
/// \param[out] destination Encoded value in CRT form
/// \param[in] he_seal_backend Backend whose context is used for encoding
/// \param[in] pool Memory pool used for new memory allocation
void encode(double value, const ngraph::element::Type& element_type,
            double scale, seal::parms_id_type parms_id,
            std::vector<std::uint64_t>& destination,
            const HESealBackend& he_seal_backend,
            seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());

/// \brief Encode value into each slot of a plaintext
/// \param[out] destination Encoded value in CRT form
/// \param[in] plaintext Input values to encode
/// \param[in] ckks_encoder Used for encoding
/// \param[in] parms_id Seal parameter id to use in encoding
/// \param[in] element_type Datatype used for encoding
/// \param[in] scale Scale at which to encode value
/// \param[in] complex_packing Whether or not to use complex packing during
/// encoding
void encode(ngraph::he::SealPlaintextWrapper& destination,
            const ngraph::he::HEPlaintext& plaintext,
            seal::CKKSEncoder& ckks_encoder, seal::parms_id_type parms_id,
            const ngraph::element::Type& element_type, double scale,
            bool complex_packing);

/// \brief Encrypt plaintext into ciphertext
/// \param[out] output Encrypted value
/// \param[in] input Plaintext to encode
/// \param[in] parms_id Seal parameter id to use in encoding
/// \param[in] element_type Datatype used for encoding
/// \param[in] scale Scale at which to encode value
/// \param[in] ckks_encoder Used for encoding
/// \param[in] encryptor Used for encrypting
/// \param[in] complex_packing Whether or not to use complex packing during
/// encoding
void encrypt(std::shared_ptr<ngraph::he::SealCiphertextWrapper>& output,
             const ngraph::he::HEPlaintext& input, seal::parms_id_type parms_id,
             const ngraph::element::Type& element_type, double scale,
             seal::CKKSEncoder& ckks_encoder, seal::Encryptor& encryptor,
             bool complex_packing);

/// \brief Decode SEAL plaintext into plaintext values
/// \param[out] output Decoded values
/// \param[in] input Plaintext to decode
/// \param[in] ckks_encoder Used for decoding
void decode(ngraph::he::HEPlaintext& output,
            const ngraph::he::SealPlaintextWrapper& input,
            seal::CKKSEncoder& ckks_encoder);

/// \brief Writes plaintext to byte output
/// \param[out] output Pointer to destination
/// \param[out] input Plaintext to write
/// \param[in] type Datatype to write
/// \param[in] count Number of values to write
void decode(void* output, const ngraph::he::HEPlaintext& input,
            const element::Type& type, size_t count);

/// \brief Decrypts and decodes a ciphertext to plaintext values
/// \param[out] output Destination to write values to
/// \param[in] input Ciphertext to decrypt
/// \param[in] decryptor Used for decryption
/// \param[in] ckks_encoder Used for decoding
void decrypt(ngraph::he::HEPlaintext& output,
             const ngraph::he::SealCiphertextWrapper& input,
             seal::Decryptor& decryptor, seal::CKKSEncoder& ckks_encoder);

}  // namespace he
}  // namespace ngraph
