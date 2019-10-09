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

#include <cstring>

#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/util.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_cipher_tensor.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"
#include "util.hpp"

namespace ngraph {
namespace he {

HESealCipherTensor::HESealCipherTensor(const element::Type& element_type,
                                       const Shape& shape,
                                       const HESealBackend& he_seal_backend,
                                       const bool packed,
                                       const std::string& name)
    : HETensor(element_type, shape, packed, name),
      m_he_seal_backend{he_seal_backend} {
  m_num_elements = m_descriptor->get_tensor_layout()->get_size() / m_batch_size;
  m_ciphertexts.resize(m_num_elements);

#pragma omp parallel for
  for (size_t i = 0; i < m_num_elements; ++i) {
    m_ciphertexts[i] = he_seal_backend.create_empty_ciphertext();
  }
}

void HESealCipherTensor::write(const void* source, size_t n) {
  check_io_bounds(source, n / m_batch_size);
  HESealCipherTensor::write(
      m_ciphertexts, source, n, m_batch_size,
      get_tensor_layout()->get_element_type(),
      m_he_seal_backend.get_context()->first_parms_id(),
      m_he_seal_backend.get_scale(), *m_he_seal_backend.get_ckks_encoder(),
      *m_he_seal_backend.get_encryptor(), m_he_seal_backend.complex_packing());
}

void HESealCipherTensor::write(
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& destination,
    const void* source, size_t n, size_t batch_size,
    const element::Type& element_type, seal::parms_id_type parms_id,
    double scale, seal::CKKSEncoder& ckks_encoder, seal::Encryptor& encryptor,
    bool complex_packing) {
  size_t type_byte_size = element_type.size();
  size_t num_elements_to_write = n / (type_byte_size * batch_size);
  NGRAPH_CHECK(destination.size() >= num_elements_to_write,
               "Writing too many ciphertexts ", num_elements_to_write,
               " to destination size ", destination.size());

  if (num_elements_to_write == 1) {
    std::vector<double> values(batch_size);
    char* src_with_offset = static_cast<char*>(const_cast<void*>(source));
    for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      values[batch_idx] =
          type_to_double(static_cast<void*>(src_with_offset), element_type);
      src_with_offset += type_byte_size;
    }
    auto plaintext = HEPlaintext(values);
    encrypt(destination[0], plaintext, parms_id, element_type, scale,
            ckks_encoder, encryptor, complex_packing);
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements_to_write; ++i) {
      auto plaintext = HEPlaintext();
      if (batch_size > 1) {
        std::vector<double> values(batch_size);
        char* src_with_offset = const_cast<char*>(
            static_cast<const char*>(source) + i * type_byte_size);
        for (size_t j = 0; j < batch_size; ++j) {
          values[j] = type_to_double(src_with_offset, element_type);
          src_with_offset += type_byte_size * num_elements_to_write;
        }
        plaintext.set_values(values);
      } else {
        std::vector<double> values(batch_size);
        char* src_with_offset = const_cast<char*>(
            static_cast<const char*>(source) + i * type_byte_size * batch_size);
        for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
          values[batch_idx] =
              type_to_double(static_cast<void*>(src_with_offset), element_type);
          src_with_offset += type_byte_size;
        }
        plaintext.set_values(values);
      }
      encrypt(destination[i], plaintext, parms_id, element_type, scale,
              ckks_encoder, encryptor, complex_packing);
    }
  }
}

void HESealCipherTensor::read(void* target, size_t n) const {
  check_io_bounds(target, n / m_batch_size);
  HESealCipherTensor::read(target, m_ciphertexts, n, m_batch_size,
                           get_tensor_layout()->get_element_type(),
                           *m_he_seal_backend.get_ckks_encoder(),
                           *m_he_seal_backend.get_decryptor());
}

void HESealCipherTensor::read(
    void* target,
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& ciphertexts,
    size_t num_bytes, size_t batch_size, const element::Type& element_type,
    seal::CKKSEncoder& ckks_encoder, seal::Decryptor& decryptor) {
  size_t type_byte_size = element_type.size();
  size_t num_elements_to_read = num_bytes / (type_byte_size * batch_size);

  NGRAPH_CHECK(ciphertexts.size() >= num_elements_to_read,
               "Reading too many elements ", num_elements_to_read,
               " from ciphertext size ", ciphertexts.size());

  if (num_elements_to_read == 1) {
    void* dst_with_offset = target;
    auto p = HEPlaintext();
    auto cipher = ciphertexts[0];
    decrypt(p, *cipher, decryptor, ckks_encoder);
    decode(dst_with_offset, p, element_type, batch_size);
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements_to_read; ++i) {
      void* dst = ngraph::ngraph_malloc(type_byte_size * batch_size);
      auto cipher = ciphertexts[i];
      auto p = HEPlaintext();
      decrypt(p, *cipher, decryptor, ckks_encoder);
      decode(dst, p, element_type, batch_size);

      for (size_t j = 0; j < batch_size; ++j) {
        void* dst_with_offset =
            static_cast<void*>(static_cast<char*>(target) +
                               type_byte_size * (i + j * num_elements_to_read));
        const void* src =
            static_cast<void*>(static_cast<char*>(dst) + j * type_byte_size);
        memcpy(dst_with_offset, src, type_byte_size);
      }
      ngraph::ngraph_free(dst);
    }
  }
}

void HESealCipherTensor::set_elements(
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& elements) {
  if (elements.size() != get_element_count() / m_batch_size) {
    NGRAPH_ERR << "m_batch_size " << m_batch_size;
    NGRAPH_ERR << "get_element_count " << get_element_count();
    NGRAPH_ERR << "elements.size " << elements.size();
    throw ngraph_error("Wrong number of elements set");
  }
  m_ciphertexts = elements;
}

}  // namespace he
}  // namespace ngraph