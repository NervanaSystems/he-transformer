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

ngraph::he::HESealCipherTensor::HESealCipherTensor(
    const element::Type& element_type, const Shape& shape,
    const ngraph::he::HESealBackend& he_seal_backend, const bool packed,
    const std::string& name)
    : ngraph::he::HETensor(element_type, shape, he_seal_backend, packed, name) {
  m_num_elements = m_descriptor->get_tensor_layout()->get_size() / m_batch_size;
  m_ciphertexts.resize(m_num_elements);

#pragma omp parallel for
  for (size_t i = 0; i < m_num_elements; ++i) {
    m_ciphertexts[i] = he_seal_backend.create_empty_ciphertext();
  }

  if (m_num_elements > 0) {
    NGRAPH_INFO << "Created cipher tensor with " << m_num_elements
                << " elements, each sized "
                << ngraph::he::ciphertext_size(m_ciphertexts[0]->ciphertext());
  }
}

void ngraph::he::HESealCipherTensor::write(const void* source,
                                           size_t tensor_offset, size_t n) {
  const bool complex_packing = m_he_seal_backend.complex_packing();

  NGRAPH_CHECK(tensor_offset == 0, "Tensor offset is not zero in Cipher write");

  check_io_bounds(source, n / m_batch_size);
  const element::Type& element_type = get_tensor_layout()->get_element_type();
  NGRAPH_CHECK(element_type == element::f32,
               "CipherTensor supports float32 only");

  size_t type_byte_size = element_type.size();
  size_t num_elements_to_write = n / (type_byte_size * m_batch_size);

  if (num_elements_to_write == 1) {
    const void* src_with_offset = (void*)((char*)source);

    std::vector<float> values{(float*)src_with_offset,
                              (float*)src_with_offset + m_batch_size};
    auto plaintext = HEPlaintext(values);
    m_he_seal_backend.encrypt(m_ciphertexts[0], plaintext, complex_packing);
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements_to_write; ++i) {
      const void* src_with_offset =
          (void*)((char*)source + i * type_byte_size * m_batch_size);

      auto plaintext = HEPlaintext();
      if (m_batch_size > 1) {
        size_t allocation_size = type_byte_size * m_batch_size;
        const void* batch_src = malloc(allocation_size);
        if (!batch_src) {
          throw ngraph_error("Error allocating HE Cipher Tensor View memory");
        }
        for (size_t j = 0; j < m_batch_size; ++j) {
          void* destination = (void*)((char*)batch_src + j * type_byte_size);
          const void* src =
              (void*)((char*)source +
                      type_byte_size * (i + j * num_elements_to_write));
          memcpy(destination, src, type_byte_size);
        }
        std::vector<float> values{(float*)batch_src,
                                  (float*)batch_src + m_batch_size};
        plaintext.values() = values;
        free((void*)batch_src);
      } else {
        std::vector<float> values{(float*)src_with_offset,
                                  (float*)src_with_offset + m_batch_size};
        plaintext.values() = values;
      }
      m_he_seal_backend.encrypt(m_ciphertexts[i], plaintext, complex_packing);
    }
  }
}

void ngraph::he::HESealCipherTensor::read(void* target, size_t tensor_offset,
                                          size_t n) const {
  NGRAPH_CHECK(tensor_offset == 0, "Tensor offset is not zero in Cipher read");
  check_io_bounds(target, n / m_batch_size);
  const element::Type& element_type = get_tensor_layout()->get_element_type();
  size_t type_byte_size = element_type.size();
  size_t num_elements_to_read = n / (type_byte_size * m_batch_size);

  if (num_elements_to_read == 1) {
    void* dst_with_offset = (void*)((char*)target);
    auto p = HEPlaintext();
    auto cipher = m_ciphertexts[0];
    m_he_seal_backend.decrypt(p, *cipher);
    m_he_seal_backend.decode(dst_with_offset, p, element_type, m_batch_size);
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements_to_read; ++i) {
      void* dst = malloc(type_byte_size * m_batch_size);
      if (!dst) {
        throw ngraph_error("Error allocating HE Cipher Tensor memory");
      }
      auto cipher = m_ciphertexts[i];
      auto p = HEPlaintext();
      m_he_seal_backend.decrypt(p, *cipher);
      m_he_seal_backend.decode(dst, p, element_type, m_batch_size);

      for (size_t j = 0; j < m_batch_size; ++j) {
        void* dst_with_offset =
            (void*)((char*)target +
                    type_byte_size * (i + j * num_elements_to_read));
        const void* src = (void*)((char*)dst + j * type_byte_size);
        memcpy(dst_with_offset, src, type_byte_size);
      }
      free(dst);
    }
  }
}

void ngraph::he::HESealCipherTensor::set_elements(
    const std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>&
        elements) {
  if (elements.size() != get_element_count() / m_batch_size) {
    NGRAPH_INFO << "m_batch_size " << m_batch_size;
    NGRAPH_INFO << "get_element_count " << get_element_count();
    NGRAPH_INFO << "elements.size " << elements.size();
    throw ngraph_error("Wrong number of elements set");
  }
  m_ciphertexts = elements;
}
