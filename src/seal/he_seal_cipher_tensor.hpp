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
#include <string>
#include <vector>

#include "he_tensor.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph {
namespace he {
/// \brief Class representing a tensor of ciphertexts
class HESealCipherTensor : public HETensor {
 public:
  /// \brief Constructs a tensor of ciphertexts
  /// \param[in] element_type Datatype of underlying cipher tensor
  /// \param[in] shape Shape of underlying cipher tensor
  /// \param[in] he_seal_backend Backend used to create the tensor
  /// \param[in] packed Whether or not to use plaintext packing
  /// \param[in] name Name of the tensor
  HESealCipherTensor(const element::Type& element_type, const Shape& shape,
                     const HESealBackend& he_seal_backend,
                     const bool packed = false,
                     const std::string& name = "external");

  /// \brief Write bytes directly into the tensor after encoding and encrypting
  /// \param[in] p Pointer to source of data
  /// \param[in] n Number of bytes to write, must be integral number of elements
  void write(const void* p, size_t n) override;

  /// \brief Write bytes into a vector of ciphertexts after encoding and
  /// encrypting
  /// \param[out] destination Ciphertexts to write to \param[in]
  /// source Pointer to source of data \param[in] n Number of bytes to write
  /// \param[in] batch_size Packing factor to use
  /// \param[in] element_type Datatype of source data
  /// \param[in] parms_id Seal parameter id to use in encryption
  /// \param[in] scale Scale at which to encode ciphertexts
  /// \param[in] ckks_encoder Encoder used for encoding
  /// \param[in] encryptor Encryptor used for encryption
  /// \param[in] complex_packing Whether or not to encode elements using complex
  /// packing
  static void write(
      std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>&
          destination,
      const void* source, size_t n, size_t batch_size,
      const element::Type& element_type, seal::parms_id_type parms_id,
      double scale, seal::CKKSEncoder& ckks_encoder, seal::Encryptor& encryptor,
      bool complex_packing);

  /// \brief Read bytes directly from the tensor after decrypting and decoding
  /// \param[in] target Pointer to destination for data
  /// \param[in] n Number of bytes to read, must be a multiple of batch size
  void read(void* target, size_t n) const override;

  /// \brief Read bytes from a vector of ciphertexts after decrpyting and
  /// decoding \param[out] target Pointer to destination for data \param[in]
  /// ciphertexts Ciphertexts to decrypt and decode \param[in] n Number of bytes
  /// to read, must be a multiple of the batch size \param[in] batch_size
  /// Packing factor to use \param[in] element_type Datatype of source data
  /// \param[in] parms_id TODO: remove
  /// \param[in] scale TODO: remove
  /// \param[in] ckks_encoder Encoder used for decoding
  /// \param[in] decryptor Decryptor used for decrpytion
  /// \param[in] complex_packing TODO: check if used
  static void read(
      void* target,
      const std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>&
          ciphertexts,
      size_t n, size_t batch_size, const element::Type& element_type,
      seal::parms_id_type parms_id, double scale,
      seal::CKKSEncoder& ckks_encoder, seal::Decryptor& decryptor,
      bool complex_packing);

  /// \brief Replaces the ciphertexts in the tensor
  /// \param[in] elements Ciphertexts to store in the tensor
  /// \throws ngraph_error if incorrect number of ciphertexts is used
  void set_elements(
      const std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>&
          elements);

  /// \brief Writes the encrypted ciphertexts to a stream
  /// \param[in,out] stream Stream to which ciphertexts are written
  void save_elements(std::ostream& stream) const {
    NGRAPH_CHECK(m_ciphertexts.size() > 0, "Cannot save 0 ciphertexts");

    size_t cipher_size = m_ciphertexts[0]->size();
    for (auto& ciphertext : m_ciphertexts) {
      NGRAPH_CHECK(cipher_size == ciphertext->size(), "Cipher size ",
                   ciphertext->size(), " doesn't match expected ", cipher_size);

      if (ciphertext->known_value()) {
        throw ngraph_error("Can't save known-valued ciphertext");
      }
      ciphertext->save(stream);
    }
  }

  /// \brief Returns the ciphertexts stored in the tensor
  inline std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>&
  get_elements() {
    return m_ciphertexts;
  }

  /// \brief Returns the ciphertext stored at a specific index in the tensor
  /// \param[in] index Index at which to return the ciphertext
  /// \throws ngraph_error if index is out of bounds
  inline std::shared_ptr<ngraph::he::SealCiphertextWrapper>& get_element(
      size_t index) {
    NGRAPH_CHECK(index < m_ciphertexts.size(), "Index ", index,
                 " out of bounds for vector of size ", m_ciphertexts.size());
    return m_ciphertexts[index];
  }

  /// \brief Returns the number of cipphertexts in the tensor
  inline size_t num_ciphertexts() { return m_ciphertexts.size(); }

  /// \brief Returns type information about the tensor
  /// /returns a reference to a HETensorTypeInfo object
  const HETensorTypeInfo& get_type_info() const override { return type_info; }

  /// \brief Represents a HESealCipherTensor type
  static constexpr HETensorTypeInfo type_info{HETensorTypeInfo::cipher};

 private:
  std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>> m_ciphertexts;
  size_t m_num_elements;
  const HESealBackend& m_he_seal_backend;
};
}  // namespace he
}  // namespace ngraph
