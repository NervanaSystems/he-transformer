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

#include "he_tensor.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/util.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {

HETensor::HETensor(const element::Type& element_type, const Shape& shape,
                   const bool plaintext_packing, const bool complex_packing,
                   const bool encrypted, const HESealBackend& he_seal_backend,
                   const std::string& name)
    : ngraph::runtime::Tensor(std::make_shared<ngraph::descriptor::Tensor>(
          element_type, shape, name)),
      m_packed(plaintext_packing),
      m_he_seal_backend(he_seal_backend) {
  m_descriptor->set_tensor_layout(
      std::make_shared<ngraph::descriptor::layout::DenseTensorLayout>(
          *m_descriptor));

  if (plaintext_packing) {
    m_packed_shape = pack_shape(shape, 0);
  } else {
    m_packed_shape = shape;
  }

  size_t num_elements =
      m_descriptor->get_tensor_layout()->get_size() / get_batch_size();

  if (encrypted) {
    m_data.resize(num_elements,
                  HEType(HESealBackend::create_empty_ciphertext(),
                         plaintext_packing, complex_packing, get_batch_size()));
    for (size_t i = 0; i < num_elements; ++i) {
      m_data[i] = HEType(HESealBackend::create_empty_ciphertext(),
                         plaintext_packing, complex_packing, get_batch_size());
    }

  } else {
    m_data.resize(num_elements, HEType(HEPlaintext(), plaintext_packing,
                                       complex_packing, get_batch_size()));
  }
}

ngraph::Shape HETensor::pack_shape(const ngraph::Shape& shape,
                                   size_t batch_axis) {
  if (batch_axis != 0) {
    throw ngraph::ngraph_error("Packing only supported along axis 0");
  }
  ngraph::Shape packed_shape(shape);
  if (shape.size() > 0 && shape[0] != 0) {
    packed_shape[0] = 1;
  }
  return packed_shape;
}

ngraph::Shape HETensor::unpack_shape(const ngraph::Shape& shape,
                                     size_t pack_size, size_t batch_axis) {
  if (batch_axis != 0) {
    throw ngraph::ngraph_error("Unpacking only supported along axis 0");
  }
  ngraph::Shape unpacked_shape(shape);
  if (shape.size() > 0 && shape[0] != 0) {
    unpacked_shape[0] = pack_size;
  }
  return unpacked_shape;
}

uint64_t HETensor::batch_size(const Shape& shape, const bool packed) {
  if (shape.size() > 0 && packed) {
    return shape[0];
  }
  return 1;
}

void HETensor::check_io_bounds(const void* p, size_t n) const {
  const element::Type& element_type = get_tensor_layout()->get_element_type();
  size_t type_byte_size = element_type.size();

  // Memory must be byte-aligned to type_byte_size
  if (n % type_byte_size != 0) {
    throw ngraph::ngraph_error("n must be divisible by type_byte_size.");
  }
  // Check out-of-range
  if (n / type_byte_size > get_element_count()) {
    throw std::out_of_range("I/O access past end of tensor");
  }
}

void HETensor::write(const void* p, size_t n) {
  check_io_bounds(p, n / get_batch_size());
  const element::Type& element_type = get_tensor_layout()->get_element_type();
  size_t type_byte_size = element_type.size();
  size_t num_elements_to_write = n / (element_type.size() * get_batch_size());
  NGRAPH_INFO << "Writing " << num_elements_to_write << " elements";

#pragma omp parallel for
  for (size_t i = 0; i < num_elements_to_write; ++i) {
    std::vector<double> values(get_batch_size());
    for (size_t j = 0; j < get_batch_size(); ++j) {
      const void* src = static_cast<const void*>(
          static_cast<const char*>(p) +
          type_byte_size * (i + j * num_elements_to_write));
      values[j] = type_to_double(src, element_type);
    }

    HEPlaintext plain({values});
    if (m_data[i].is_plaintext()) {
      m_data[i].set_plaintext(plain);
      NGRAPH_INFO << "Setting elemnt " << plain;
    } else if (m_data[i].is_ciphertext()) {
      NGRAPH_INFO << "Encrypting value " << plain;
      NGRAPH_INFO << "Complex packing? " << m_data[i].complex_packing();
      auto cipher = HESealBackend::create_empty_ciphertext();
      m_he_seal_backend.encrypt(cipher, plain, element_type,
                                m_data[i].complex_packing());
      m_data[i].set_ciphertext(cipher);
    } else {
      NGRAPH_CHECK(false, "Cannot write into tensor of unspecified type");
    }
  }
}

void HETensor::read(void* target, size_t n) const {
  check_io_bounds(target, n);
  const element::Type& element_type = get_tensor_layout()->get_element_type();
  size_t type_byte_size = element_type.size();
  size_t num_elements_to_read = n / (type_byte_size * get_batch_size());

  NGRAPH_INFO << "Reading " << num_elements_to_read << " elements (batch size "
              << get_batch_size();

  auto copy_batch_values_to_src = [&](size_t element_idx, void* copy_target,
                                      const void* type_values_src) {
    char* src = static_cast<char*>(const_cast<void*>(type_values_src));
    for (size_t j = 0; j < get_batch_size(); ++j) {
      void* dst_with_offset = static_cast<void*>(
          static_cast<char*>(copy_target) +
          type_byte_size * (element_idx + j * num_elements_to_read));
      std::memcpy(dst_with_offset, src, type_byte_size);
      src += type_byte_size;
    }
  };

#pragma omp parallel for
  for (size_t i = 0; i < num_elements_to_read; ++i) {
    NGRAPH_CHECK(m_data[i].is_plaintext() || m_data[i].is_ciphertext(),
                 "Cannot read from tensor of unspecified type");

    HEPlaintext plain;
    if (m_data[i].is_ciphertext()) {
      auto cipher = m_data[i].get_ciphertext();
      m_he_seal_backend.decrypt(plain, *cipher, m_data[i].complex_packing());
    } else {
      plain = m_data[i].get_plaintext();
    }

    void* dst = ngraph::ngraph_malloc(type_byte_size * get_batch_size());
    ngraph::he::decode(dst, plain, element_type, get_batch_size());

    copy_batch_values_to_src(i, target, dst);
    ngraph::ngraph_free(dst);
  }
}

}  // namespace he
}  // namespace ngraph