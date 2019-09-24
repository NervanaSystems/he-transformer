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
  /// \param[in] shape Shape of underyling cipher tensor
  /// \param[in] he_seal_backend Backend own
  HESealCipherTensor(const element::Type& element_type, const Shape& shape,
                     const HESealBackend& he_seal_backend,
                     const bool packed = false,
                     const std::string& name = "external");

  /// \brief Write bytes directly into the tensor after encoding and encrypting
  /// \param p Pointer to source of data
  /// \param n Number of bytes to write, must be integral number of elements
  void write(const void* p, size_t n) override;

  /// \brief
  static void write(
      std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>&
          destination,
      const void* source, size_t n, size_t batch_size,
      const element::Type& element_type, seal::parms_id_type parms_id,
      double scale, seal::CKKSEncoder& ckks_encoder, seal::Encryptor& encryptor,
      bool complex_packing);

  /// \brief Read bytes directly from the tensor after decrypting and decoding
  /// \param p Pointer to destination for data
  /// \param n Number of bytes to read, must be integral number of elements
  void read(void* target, size_t n) const override;

  static void read(
      void* target,
      const std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>&
          ciphertexts,
      size_t n, size_t batch_size, const element::Type& element_type,
      seal::parms_id_type parms_id, double scale,
      seal::CKKSEncoder& ckks_encoder, seal::Decryptor& decryptor,
      bool complex_packing);

  void set_elements(
      const std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>&
          elements);

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

  inline std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>&
  get_elements() {
    return m_ciphertexts;
  }

  inline std::shared_ptr<ngraph::he::SealCiphertextWrapper>& get_element(
      size_t i) {
    NGRAPH_CHECK(i < m_ciphertexts.size(), "Index ", i,
                 " out of bounds for vector of size ", m_ciphertexts.size());
    return m_ciphertexts[i];
  }

  inline size_t num_ciphertexts() { return m_ciphertexts.size(); }

  const HETensorTypeInfo& get_type_info() const override { return type_info; }

  static constexpr HETensorTypeInfo type_info{HETensorTypeInfo::cipher};

 private:
  std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>> m_ciphertexts;
  size_t m_num_elements;
};
}  // namespace he
}  // namespace ngraph
