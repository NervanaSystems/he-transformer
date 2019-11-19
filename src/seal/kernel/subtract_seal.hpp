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

#include <memory>
#include <vector>

#include "he_type.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/negate_seal.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph::runtime::he {
/// \brief Subtracts two ciphertexts
/// \param[in,out] arg0 Ciphertext argument to subtract from. May be rescaled
/// \param[in,out] arg1 Ciphertext argument to subtract. May be rescaled
/// \param[out] out Stores the encrypted sum
/// \param[in] he_seal_backend Backend with which to perform subtractition
/// \param[in] pool Memory pool used for new memory allocation
void scalar_subtract_seal(
    SealCiphertextWrapper& arg0, SealCiphertextWrapper& arg1,
    std::shared_ptr<SealCiphertextWrapper>& out, HESealBackend& he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

/// \brief Subtracts a ciphertext by a plaintext
/// \param[in,out] arg0 Ciphertext argument to subtract from. May be rescaled
/// \param[in] arg1 Plaintext argument to subtract
/// \param[out] out Stores the encrypted sum
/// \param[in] complex_packing Whether or not the ciphertext should be
/// multiplied using complex packing
/// \param[in] he_seal_backend Backend with which to perform subtractition
void scalar_subtract_seal(SealCiphertextWrapper& arg0, const HEPlaintext& arg1,
                          std::shared_ptr<SealCiphertextWrapper>& out,
                          const bool complex_packing,
                          HESealBackend& he_seal_backend);

/// \brief Subtracts a plaintext by a ciphertext
/// \param[in] arg0 Plaintext argument to subtract from
/// \param[in,out] arg1 Ciphertext argument to subtract. May be rescaled
/// \param[out] out Stores the encrypted sum
/// \param[in] complex_packing Whether or not the ciphertext should be
/// multiplied using complex packing
/// \param[in] he_seal_backend Backend with which to perform subtractition
void scalar_subtract_seal(const HEPlaintext& arg0, SealCiphertextWrapper& arg1,
                          std::shared_ptr<SealCiphertextWrapper>& out,
                          const bool complex_packing,
                          HESealBackend& he_seal_backend);

/// \brief Subtracts two plaintexts
/// \param[in] arg0 Plaintext argument to subtract from
/// \param[in] arg1 Plaintext argument to subtract
/// \param[out] out Stores the plaintext difference
void scalar_subtract_seal(const HEPlaintext& arg0, const HEPlaintext& arg1,
                          HEPlaintext& out);

/// \brief Subtracts two ciphertext/plaintext elements
/// \param[in] arg0 Cipher or plaintext data to subtract from
/// \param[in] arg1 Cipher or plaintext data to subtract
/// \param[in] out Stores the ciphertext or plaintext difference
/// \param[in] he_seal_backend Backend used to perform subtraction
inline void scalar_subtract_seal(HEType& arg0, HEType& arg1, HEType& out,
                                 HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(arg0.complex_packing() == arg1.complex_packing(),
               "Complex packing types don't match");
  out.complex_packing() = arg0.complex_packing();

  if (arg0.is_ciphertext() && arg1.is_ciphertext()) {
    scalar_subtract_seal(*arg0.get_ciphertext(), *arg1.get_ciphertext(),
                         out.get_ciphertext(), he_seal_backend);
  } else if (arg0.is_ciphertext() && arg1.is_plaintext()) {
    scalar_subtract_seal(*arg0.get_ciphertext(), arg1.get_plaintext(),
                         out.get_ciphertext(), arg0.complex_packing(),
                         he_seal_backend);
  } else if (arg0.is_plaintext() && arg1.is_ciphertext()) {
    scalar_subtract_seal(arg0.get_plaintext(), *arg1.get_ciphertext(),
                         out.get_ciphertext(), arg0.complex_packing(),
                         he_seal_backend);
  } else if (arg0.is_plaintext() && arg1.is_plaintext()) {
    scalar_subtract_seal(arg0.get_plaintext(), arg1.get_plaintext(),
                         out.get_plaintext());
  }
}

/// \brief Subtracts two vectors of ciphertext/plaintext elements element-wise
/// \param[in] arg0 Cipher or plaintext data to subtract from
/// \param[in] arg1 Cipher or plaintext data to subtract
/// \param[in] out Stores the ciphertext or plaintext difference
/// \param[in] count Number of elements to subtract
/// \param[in] element_type datatype of the data to subtract
/// \param[in] he_seal_backend Backend used to perform multiplication
inline void subtract_seal(std::vector<HEType>& arg0, std::vector<HEType>& arg1,
                          std::vector<HEType>& out, size_t count,
                          const element::Type& element_type,
                          HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
  NGRAPH_CHECK(count <= arg0.size(), "Count ", count,
               " is too large for arg0, with size ", arg0.size());
  NGRAPH_CHECK(count <= arg1.size(), "Count ", count,
               " is too large for arg1, with size ", arg1.size());

#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_subtract_seal(arg0[i], arg1[i], out[i], he_seal_backend);
  }
}

}  // namespace ngraph::runtime::he
