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
#include "util.hpp"

namespace ngraph {
namespace he {

HEPlainTensor::HEPlainTensor(const element::Type& element_type,
                             const Shape& shape, const bool packed,
                             const std::string& name)
    : HETensor(element_type, shape, packed, name) {
  size_t num_elements =
      m_descriptor->get_tensor_layout()->get_size() / m_batch_size;
  m_plaintexts.resize(num_elements);
}

void HEPlainTensor::write(const void* source, size_t n) {
  check_io_bounds(source, n / m_batch_size);
  const element::Type& element_type = get_tensor_layout()->get_element_type();
  size_t type_byte_size = element_type.size();
  size_t num_elements_to_write = n / (element_type.size() * m_batch_size);

  if (num_elements_to_write == 1) {
    const void* src_with_offset = source;
    if (m_batch_size > 1 && is_packed()) {
      std::vector<double> values(m_batch_size);

      for (size_t j = 0; j < m_batch_size; ++j) {
        const void* src = static_cast<const void*>(
            static_cast<const char*>(source) +
            type_byte_size * (j * num_elements_to_write));

        values[j] = type_to_double(src, element_type);
      }
      m_plaintexts[0].set_values(values);

    } else {
      const double d = type_to_double(src_with_offset, element_type);
      m_plaintexts[0].set_value(d);
    }
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements_to_write; ++i) {
      const void* src_with_offset = static_cast<const void*>(
          static_cast<const char*>(source) + i * type_byte_size);
      if (m_batch_size > 1) {
        std::vector<double> values(m_batch_size);

        for (size_t j = 0; j < m_batch_size; ++j) {
          const void* src = static_cast<const void*>(
              static_cast<const char*>(source) +
              type_byte_size * (i + j * num_elements_to_write));

          values[j] = type_to_double(src, element_type);
        }
        m_plaintexts[i].set_values(values);
      } else {
        const double d = type_to_double(src_with_offset, element_type);
        m_plaintexts[i].set_value(d);
      }
    }
  }
}

void HEPlainTensor::read(void* target, size_t n) const {
  check_io_bounds(target, n);
  const element::Type& element_type = get_tensor_layout()->get_element_type();

  size_t type_byte_size = element_type.size();
  size_t num_elements_to_read = n / (type_byte_size * m_batch_size);

  if (num_elements_to_read == 1) {
    void* dst_with_offset = target;
    NGRAPH_CHECK(m_plaintexts.size() > 0,
                 "Cannot read from empty plain tensor");
    const std::vector<double>& values = m_plaintexts[0].values();
    NGRAPH_CHECK(values.size() > 0, "Cannot read from empty plaintext");

    switch (element_type.get_type_enum()) {
      case element::Type_t::f32: {
        std::vector<float> float_values{values.begin(), values.end()};
       const void* type_values_src =
            static_cast<const void*>(float_values.data());
        std::memcpy(dst_with_offset, type_values_src,
                    type_byte_size * m_batch_size);
        break;
      }
      case element::Type_t::f64: {
        const void* type_values_src =
            static_cast<const void*>(values.data());
        std::memcpy(dst_with_offset, type_values_src,
                    type_byte_size * m_batch_size);
        break;
      }
      case element::Type_t::i64: {
        std::vector<int64_t> int64_values{values.begin(), values.end()};
        const void* type_values_src =
            static_cast<const void*>(int64_values.data());
        std::memcpy(dst_with_offset, type_values_src,
                    type_byte_size * m_batch_size);
        break;
      }
      case element::Type_t::i8:
      case element::Type_t::i16:
      case element::Type_t::i32:
      case element::Type_t::u8:
      case element::Type_t::u16:
      case element::Type_t::u32:
      case element::Type_t::u64:
      case element::Type_t::dynamic:
      case element::Type_t::undefined:
      case element::Type_t::bf16:
      case element::Type_t::f16:
      case element::Type_t::boolean:
        NGRAPH_CHECK(false, "Unsupported element type ", element_type);
        break;
    }

  } else {
    auto copy_batch_values_to_src = [&](size_t element_idx, void* copy_target,
                                        const void* type_values_src) {
      char* src = static_cast<char*>(const_cast<void*>(type_values_src));
      for (size_t j = 0; j < m_batch_size; ++j) {
        void* dst_with_offset = static_cast<void*>(
            static_cast<char*>(copy_target) +
            type_byte_size * (element_idx + j * num_elements_to_read));
        std::memcpy(dst_with_offset, src, type_byte_size);
        src += type_byte_size;
      }
    };

#pragma omp parallel for
    for (size_t i = 0; i < num_elements_to_read; ++i) {
      const std::vector<double>& values = m_plaintexts[i].values();
      NGRAPH_CHECK(values.size() >= m_batch_size, "values size ", values.size(),
                   " is smaller than batch size ", m_batch_size);

      switch (element_type.get_type_enum()) {
        case element::Type_t::f32: {
          std::vector<float> float_values{values.begin(), values.end()};
          const void* type_values_src =
              static_cast<const void*>(float_values.data());
          copy_batch_values_to_src(i, target, type_values_src);
          break;
        }
        case element::Type_t::f64: {
          const void* type_values_src =
              static_cast<const void*>(values.data());
          copy_batch_values_to_src(i, target, type_values_src);
          break;
        }
        case element::Type_t::i32: {
          std::vector<int32_t> int32_values{values.begin(), values.end()};
          const void* type_values_src =
              static_cast<const void*>(int32_values.data());
          copy_batch_values_to_src(i, target, type_values_src);
          break;
        }
        case element::Type_t::i64: {
          std::vector<int64_t> int64_values{values.begin(), values.end()};
          const void* type_values_src =
              static_cast<const void*>(int64_values.data());
          copy_batch_values_to_src(i, target, type_values_src);
          break;
        }
        case element::Type_t::i8:
        case element::Type_t::i16:

        case element::Type_t::u8:
        case element::Type_t::u16:
        case element::Type_t::u32:
        case element::Type_t::u64:
        case element::Type_t::dynamic:
        case element::Type_t::undefined:
        case element::Type_t::bf16:
        case element::Type_t::f16:
        case element::Type_t::boolean:
          NGRAPH_CHECK(false, "Unsupported element type ", element_type);
          break;
      }
    }
  }
}

void HEPlainTensor::set_elements(const std::vector<HEPlaintext>& elements) {
  if (elements.size() != get_element_count() / m_batch_size) {
    NGRAPH_ERR << "m_batch_size " << m_batch_size;
    NGRAPH_ERR << "get_element_count " << get_element_count();
    NGRAPH_ERR << "elements.size " << elements.size();
    throw ngraph_error("Wrong number of elements set");
  }
  m_plaintexts = elements;
}

void HEPlainTensor::pack() {
  if (is_packed()) {
    return;
  }

  m_batch_size = HETensor::batch_size(get_shape(), true);

  std::vector<HEPlaintext> new_plaintexts(num_plaintexts() / m_batch_size);
  for (size_t i = 0; i < new_plaintexts.size(); ++i) {
    new_plaintexts[i] = HEPlaintext();
  }

  for (size_t plain_idx = 0; plain_idx < m_plaintexts.size(); ++plain_idx) {
    auto& values = m_plaintexts[plain_idx].values();
    if (values.size() != 0) {
      size_t new_plain_idx = plain_idx % new_plaintexts.size();
      new_plaintexts[new_plain_idx].values().emplace_back(values[0]);
    }
  }

  m_plaintexts = std::move(new_plaintexts);
  m_packed = true;
  m_packed_shape = ngraph::he::HETensor::pack_shape(get_shape());
}

void HEPlainTensor::unpack() {
  if (!is_packed()) {
    return;
  }
  size_t batch_size = get_batch_size();

  std::vector<HEPlaintext> new_plaintexts;
  for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    for (size_t i = 0; i < m_plaintexts.size(); ++i) {
      auto vals = m_plaintexts[i].values();
      new_plaintexts.emplace_back(
          HEPlaintext(m_plaintexts[i].values()[batch_idx]));
    }
  }

  m_plaintexts = std::move(new_plaintexts);
  m_packed = false;
  m_batch_size = 1;
  m_packed_shape = ngraph::he::HETensor::unpack_shape(get_shape(), batch_size);
}

}  // namespace he
}  // namespace ngraph