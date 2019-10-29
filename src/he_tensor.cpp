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

HETensor::HETensor(
    const element::Type& element_type, const Shape& shape,
    const bool plaintext_packing, const bool complex_packing,
    const bool encrypted, seal::CKKSEncoder& ckks_encoder,
    std::shared_ptr<seal::SEALContext> context,
    const seal::Encryptor& encryptor, seal::Decryptor& decryptor,
    const ngraph::he::HESealEncryptionParameters& encryption_params,
    const std::string& name)
    : ngraph::runtime::Tensor(std::make_shared<ngraph::descriptor::Tensor>(
          element_type, shape, name)),
      m_packed(plaintext_packing),
      m_ckks_encoder(ckks_encoder),
      m_context(std::move(context)),
      m_encryptor(encryptor),
      m_decryptor(decryptor),
      m_encryption_params(encryption_params) {
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
    m_data.resize(num_elements, HEType(HESealBackend::create_empty_ciphertext(),
                                       complex_packing, get_batch_size()));
    for (size_t i = 0; i < num_elements; ++i) {
      m_data[i] = HEType(HESealBackend::create_empty_ciphertext(),
                         complex_packing, get_batch_size());
    }

  } else {
    m_data.resize(num_elements,
                  HEType(HEPlaintext(get_batch_size()), complex_packing));
  }
}

HETensor::HETensor(const element::Type& element_type, const Shape& shape,
                   const bool plaintext_packing, const bool complex_packing,
                   const bool encrypted, const HESealBackend& he_seal_backend,
                   const std::string& name)
    : HETensor(element_type, shape, plaintext_packing, complex_packing,
               encrypted, *he_seal_backend.get_ckks_encoder(),
               he_seal_backend.get_context(), *he_seal_backend.get_encryptor(),
               *he_seal_backend.get_decryptor(),
               he_seal_backend.get_encryption_parameters(), name) {}

ngraph::Shape HETensor::pack_shape(const ngraph::Shape& shape,
                                   size_t pack_axis) {
  if (pack_axis != 0) {
    throw ngraph::ngraph_error("Packing only supported along axis 0");
  }
  ngraph::Shape packed_shape(shape);
  if (!shape.empty() && shape[0] != 0) {
    packed_shape[0] = 1;
  }
  return packed_shape;
}

void HETensor::unpack_shape(ngraph::Shape& shape, size_t pack_size,
                            size_t pack_axis) {
  if (pack_axis != 0) {
    throw ngraph::ngraph_error("Unpacking only supported along axis 0");
  }
  if (!shape.empty() && shape[0] != 0) {
    shape[0] = pack_size;
  }
}

void HETensor::pack(size_t pack_axis) {
  NGRAPH_CHECK(pack_axis == 0, "Packing only supported along axis 0");
  if (is_packed()) {
    return;
  }
  NGRAPH_CHECK(!any_encrypted_data(),
               "Packing only supported for plaintext tensors");

  m_packed = true;
  std::vector<HEType> new_data(m_data.size() / get_batch_size(),
                               HEType(HEPlaintext(), false));

  for (size_t idx = 0; idx < m_data.size(); ++idx) {
    auto& plain = m_data[idx].get_plaintext();
    if (!plain.empty()) {
      size_t new_idx = idx % new_data.size();
      new_data[new_idx].get_plaintext().emplace_back(plain[0]);
      new_data[new_idx].complex_packing() = m_data[idx].complex_packing();
    }
  }

  m_data = std::move(new_data);
  m_packed = true;
  m_packed_shape = ngraph::he::HETensor::pack_shape(get_shape());
}

void HETensor::unpack() {
  if (!is_packed()) {
    return;
  }
  NGRAPH_CHECK(!any_encrypted_data(),
               "Unpacking only supported for plaintext tensors");

  size_t old_batch_size = get_batch_size();
  m_packed = false;
  std::vector<HEType> new_data;
  for (size_t batch_idx = 0; batch_idx < old_batch_size; ++batch_idx) {
    for (auto& data : m_data) {
      auto& plain = data.get_plaintext();
      new_data.emplace_back(
          HEPlaintext({static_cast<double>(plain[batch_idx])}), false);
    }
  }
  m_data = std::move(new_data);
  m_packed_shape = get_shape();
}

uint64_t HETensor::batch_size(const Shape& shape, const bool packed) {
  if (!shape.empty() && packed) {
    return shape[0];
  }
  return 1;
}

bool HETensor::any_encrypted_data() const {
  return std::any_of(m_data.begin(), m_data.end(), [](const HEType& he_type) {
    return he_type.is_ciphertext();
  });
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

#pragma omp parallel for
  for (size_t i = 0; i < num_elements_to_write; ++i) {
    std::vector<double> values(get_batch_size());
    for (size_t j = 0; j < get_batch_size(); ++j) {
      const auto* src = static_cast<const void*>(
          static_cast<const char*>(p) +
          type_byte_size * (i + j * num_elements_to_write));
      values[j] = type_to_double(src, element_type);
    }

    HEPlaintext plain({values});
    if (m_data[i].is_plaintext()) {
      m_data[i].set_plaintext(plain);
    } else if (m_data[i].is_ciphertext()) {
      auto cipher = HESealBackend::create_empty_ciphertext();

      ngraph::he::encrypt(cipher, plain, m_context->first_parms_id(),
                          element_type, m_encryption_params.scale(),
                          m_ckks_encoder, m_encryptor,
                          m_data[i].complex_packing());
      m_data[i].set_ciphertext(cipher);
    } else {
      NGRAPH_CHECK(false, "Cannot write into tensor of unspecified type");
    }
  }
}

void HETensor::read(void* p, size_t n) const {
  check_io_bounds(p, n);
  const element::Type& element_type = get_tensor_layout()->get_element_type();
  size_t type_byte_size = element_type.size();
  size_t num_elements_to_read = n / (type_byte_size * get_batch_size());

  auto copy_batch_values_to_src = [&](size_t element_idx, void* copy_target,
                                      const void* type_values_src) {
    auto* src = static_cast<char*>(const_cast<void*>(type_values_src));
    for (size_t j = 0; j < get_batch_size(); ++j) {
      auto* dst_with_offset = static_cast<void*>(
          static_cast<char*>(copy_target) +
          type_byte_size * (element_idx + j * num_elements_to_read));
      std::memcpy(dst_with_offset, src, type_byte_size);
      src += type_byte_size;
    }
  };

#pragma omp parallel for
  for (size_t i = 0; i < num_elements_to_read; ++i) {
    HEPlaintext plain;
    if (m_data[i].is_ciphertext()) {
      ngraph::he::decrypt(plain, *m_data[i].get_ciphertext(),
                          m_data[i].complex_packing(), m_decryptor,
                          m_ckks_encoder);
    } else {
      plain = m_data[i].get_plaintext();
    }

    void* dst = ngraph::ngraph_malloc(type_byte_size * get_batch_size());
    ngraph::he::write_plaintext(dst, plain, element_type, get_batch_size());

    copy_batch_values_to_src(i, p, dst);
    ngraph::ngraph_free(dst);
  }
}

void HETensor::write_to_protos(
    std::vector<he_proto::HETensor>& proto_tensors) const {
  // TODO(fboemer): support large shapes
  proto_tensors.resize(1);
  proto_tensors[0].set_name(get_name());
  proto_tensors[0].set_packed(m_packed);
  proto_tensors[0].set_offset(0);

  std::vector<uint64_t> int_shape{get_shape()};
  *proto_tensors[0].mutable_shape() = {int_shape.begin(), int_shape.end()};

  for (const auto& he_type : m_data) {
    he_type.save(*proto_tensors[0].add_data());
  }
}

std::shared_ptr<HETensor> HETensor::load_from_proto_tensors(
    const std::vector<he_proto::HETensor>& proto_tensors,
    seal::CKKSEncoder& ckks_encoder, std::shared_ptr<seal::SEALContext> context,
    const seal::Encryptor& encryptor, seal::Decryptor& decryptor,
    const ngraph::he::HESealEncryptionParameters& encryption_params) {
  NGRAPH_CHECK(proto_tensors.size() == 1,
               "Load from protos only supports 1 proto");

  const auto& proto_tensor = proto_tensors[0];
  const auto& proto_name = proto_tensor.name();
  const auto& proto_packed = proto_tensor.packed();
  const auto& proto_shape = proto_tensor.shape();
  size_t result_count = proto_tensor.data_size();
  ngraph::Shape shape{proto_shape.begin(), proto_shape.end()};
  auto element_type = element::f64;

  auto he_tensor = std::make_shared<HETensor>(
      element_type, shape, proto_packed, encryption_params.complex_packing(),
      false, ckks_encoder, context, encryptor, decryptor, encryption_params,
      proto_name);

#pragma omp parallel for
  for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
    auto loaded = HEType::load(proto_tensor.data(result_idx), context);
    he_tensor->data(result_idx) = loaded;
  }

  return he_tensor;
}

}  // namespace he
}  // namespace ngraph
