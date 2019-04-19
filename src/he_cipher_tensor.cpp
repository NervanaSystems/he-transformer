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

#include "he_backend.hpp"
#include "he_cipher_tensor.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

runtime::he::HECipherTensor::HECipherTensor(
    const element::Type& element_type, const Shape& shape,
    HEBackend* he_backend, const std::shared_ptr<HECiphertext> he_ciphertext,
    const bool batched, const string& name)
    : runtime::he::HETensor(element_type, shape, he_backend, batched, name) {
  m_num_elements = m_descriptor->get_tensor_layout()->get_size() / m_batch_size;
  m_cipher_texts.resize(m_num_elements);
#pragma omp parallel for
  for (size_t i = 0; i < m_num_elements; ++i) {
    m_cipher_texts[i] = he_backend->create_empty_ciphertext();
  }
}

void runtime::he::HECipherTensor::write(const void* source,
                                        size_t tensor_offset, size_t n) {
  check_io_bounds(source, tensor_offset, n / m_batch_size);
  const element::Type& element_type = get_tensor_layout()->get_element_type();
  size_t type_byte_size = element_type.size();
  size_t dst_start_index = tensor_offset / type_byte_size;
  size_t num_elements_to_write = n / (type_byte_size * m_batch_size);

  if (num_elements_to_write == 1) {
    const void* src_with_offset = (void*)((char*)source);
    size_t dst_index = dst_start_index;

    shared_ptr<runtime::he::HEPlaintext> plaintext =
        m_he_backend->create_empty_plaintext();
    m_he_backend->encode(plaintext, src_with_offset, element_type,
                         m_batch_size);
    m_he_backend->encrypt(m_cipher_texts[dst_index], plaintext.get());
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements_to_write; ++i) {
      const void* src_with_offset =
          (void*)((char*)source + i * type_byte_size * m_batch_size);
      size_t dst_index = dst_start_index + i;

      shared_ptr<runtime::he::HEPlaintext> plaintext =
          m_he_backend->create_empty_plaintext();
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

        m_he_backend->encode(plaintext, batch_src, element_type, m_batch_size);
        free((void*)batch_src);
      } else {
        m_he_backend->encode(plaintext, src_with_offset, element_type,
                             m_batch_size);
      }
      m_he_backend->encrypt(m_cipher_texts[dst_index], plaintext.get());
    }
  }
}

void runtime::he::HECipherTensor::read(void* target, size_t tensor_offset,
                                       size_t n) const {
  check_io_bounds(target, tensor_offset, n / m_batch_size);
  const element::Type& element_type = get_tensor_layout()->get_element_type();
  size_t type_byte_size = element_type.size();
  size_t src_start_index = tensor_offset / type_byte_size;
  size_t num_elements_to_read = n / (type_byte_size * m_batch_size);

  if (num_elements_to_read == 1) {
    void* dst_with_offset = (void*)((char*)target);
    size_t src_index = src_start_index;
    shared_ptr<runtime::he::HEPlaintext> p =
        m_he_backend->create_empty_plaintext();
    m_he_backend->decrypt(p, m_cipher_texts[src_index].get());
    m_he_backend->decode(dst_with_offset, p.get(), element_type, m_batch_size);
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements_to_read; ++i) {
      void* dst = malloc(type_byte_size * m_batch_size);
      if (!dst) {
        throw ngraph_error("Error allocating HE Cipher Tensor memory");
      }

      size_t src_index = src_start_index + i;
      shared_ptr<runtime::he::HEPlaintext> p =
          m_he_backend->create_empty_plaintext();
      m_he_backend->decrypt(p, m_cipher_texts[src_index].get());
      m_he_backend->decode(dst, p.get(), element_type, m_batch_size);

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

void runtime::he::HECipherTensor::set_elements(
    const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& elements) {
  if (elements.size() != get_element_count() / m_batch_size) {
    NGRAPH_INFO << "m_batch_size " << m_batch_size;
    NGRAPH_INFO << "get_element_count " << get_element_count();
    NGRAPH_INFO << "elements.size " << elements.size();
    throw ngraph_error("Wrong number of elements set");
  }
  m_cipher_texts = elements;
}
