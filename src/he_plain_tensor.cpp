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

#include "he_plain_tensor.hpp"
#include "seal/he_seal_backend.hpp"

ngraph::he::HEPlainTensor::HEPlainTensor(const element::Type& element_type,
                                         const Shape& shape,
                                         const HESealBackend& he_seal_backend,
                                         const bool packed,
                                         const std::string& name)
    : ngraph::he::HETensor(element_type, shape, he_seal_backend, packed, name) {
  m_num_elements = m_descriptor->get_tensor_layout()->get_size() / m_batch_size;
  m_plaintexts.resize(m_num_elements);
}

void ngraph::he::HEPlainTensor::write(const void* source, size_t n) {
  check_io_bounds(source, n / m_batch_size);
  const element::Type& element_type = get_tensor_layout()->get_element_type();
  size_t type_byte_size = element_type.size();
  size_t num_elements_to_write = n / (type_byte_size * m_batch_size);

  if (num_elements_to_write == 1) {
    const void* src_with_offset = (void*)((char*)source);
    if (m_batch_size > 1 && is_packed()) {
      std::vector<float> values(m_batch_size);

      for (size_t j = 0; j < m_batch_size; ++j) {
        const void* src = (void*)((char*)source +
                                  type_byte_size * (j * num_elements_to_write));

        float val = *(float*)(src);
        values[j] = val;
      }
      m_plaintexts[0].values() = values;

    } else {
      float f = *(float*)(src_with_offset);
      m_plaintexts[0].values() = {f};
    }
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements_to_write; ++i) {
      const void* src_with_offset = (void*)((char*)source + i * type_byte_size);
      if (m_batch_size > 1) {
        std::vector<float> values(m_batch_size);

        for (size_t j = 0; j < m_batch_size; ++j) {
          const void* src =
              (void*)((char*)source +
                      type_byte_size * (i + j * num_elements_to_write));

          float val = *(float*)(src);
          values[j] = val;
        }
        m_plaintexts[i].values() = values;
      } else {
        float f = *(float*)(src_with_offset);
        m_plaintexts[i].values() = {f};
      }
    }
  }
}

void ngraph::he::HEPlainTensor::read(void* target, size_t n) const {
  check_io_bounds(target, n);
  const element::Type& element_type = get_tensor_layout()->get_element_type();
  NGRAPH_CHECK(element_type == element::f32, "Only support float32");
  size_t type_byte_size = element_type.size();
  size_t num_elements_to_read = n / (type_byte_size * m_batch_size);

  if (num_elements_to_read == 1) {
    void* dst_with_offset = (void*)((char*)target);
    const std::vector<float>& values = m_plaintexts[0].values();
    NGRAPH_CHECK(values.size() > 0, "Cannot read from empty plaintext");
    memcpy(dst_with_offset, &values[0], type_byte_size * m_batch_size);
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements_to_read; ++i) {
      const std::vector<float>& values = m_plaintexts[i].values();
      NGRAPH_CHECK(values.size() >= m_batch_size, "values size ", values.size(),
                   " is smaller than batch size ", m_batch_size);

      for (size_t j = 0; j < m_batch_size; ++j) {
        void* dst_with_offset =
            (void*)((char*)target +
                    type_byte_size * (i + j * num_elements_to_read));
        const void* src = (void*)(&values[j]);
        memcpy(dst_with_offset, src, type_byte_size);
      }
    }
  }
}

void ngraph::he::HEPlainTensor::set_elements(
    const std::vector<ngraph::he::HEPlaintext>& elements) {
  if (elements.size() != get_element_count() / m_batch_size) {
    NGRAPH_INFO << "m_batch_size " << m_batch_size;
    NGRAPH_INFO << "get_element_count " << get_element_count();
    NGRAPH_INFO << "elements.size " << elements.size();
    throw ngraph_error("Wrong number of elements set");
  }
  m_plaintexts = elements;
}
