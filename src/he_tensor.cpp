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

#include <limits>

#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/util.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_util.hpp"

namespace ngraph::runtime::he {
HETensor::HETensor(const element::Type& element_type, const Shape& shape,
                   bool plaintext_packing, bool complex_packing, bool encrypted,
                   seal::CKKSEncoder& ckks_encoder,
                   std::shared_ptr<seal::SEALContext> context,
                   const seal::Encryptor& encryptor, seal::Decryptor& decryptor,
                   const HESealEncryptionParameters& encryption_params,
                   const std::string& name)
    : runtime::Tensor(
          std::make_shared<descriptor::Tensor>(element_type, shape, name)),
      m_packed(plaintext_packing),
      m_ckks_encoder(ckks_encoder),
      m_context(std::move(context)),
      m_encryptor(encryptor),
      m_decryptor(decryptor),
      m_encryption_params(encryption_params) {
  m_descriptor->set_tensor_layout(
      std::make_shared<descriptor::layout::DenseTensorLayout>(*m_descriptor));

  if (plaintext_packing) {
    m_packed_shape = pack_shape(shape, 0);
  } else {
    m_packed_shape = shape;
  }

  size_t num_elements =
      (get_batch_size() != 0)
          ? m_descriptor->get_tensor_layout()->get_size() / get_batch_size()
          : 0;

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
                   bool plaintext_packing, bool complex_packing, bool encrypted,
                   const HESealBackend& he_seal_backend,
                   const std::string& name)
    : HETensor(element_type, shape, plaintext_packing, complex_packing,
               encrypted, *he_seal_backend.get_ckks_encoder(),
               he_seal_backend.get_context(), *he_seal_backend.get_encryptor(),
               *he_seal_backend.get_decryptor(),
               he_seal_backend.get_encryption_parameters(), name) {}

Shape HETensor::pack_shape(const Shape& shape, size_t pack_axis) {
  if (pack_axis != 0) {
    throw ngraph_error("Packing only supported along axis 0");
  }
  Shape packed_shape(shape);
  if (!shape.empty() && shape[0] != 0) {
    packed_shape[0] = 1;
  }
  return packed_shape;
}

Shape HETensor::unpack_shape(const Shape& shape, size_t pack_size,
                             size_t pack_axis) {
  Shape unpacked_shape(shape);

  if (pack_axis != 0) {
    throw ngraph_error("Unpacking only supported along axis 0");
  }
  if (!shape.empty() && shape[0] != 0) {
    unpacked_shape[0] = pack_size;
  }
  return unpacked_shape;
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
  std::vector<HEPlaintext> new_plaintexts(new_data.size());

  for (size_t idx = 0; idx < m_data.size(); ++idx) {
    auto& plain = m_data[idx].get_plaintext();
    if (!plain.empty()) {
      size_t new_idx = idx % new_data.size();
      new_plaintexts[new_idx].emplace_back(plain[0]);
      new_data[new_idx].complex_packing() = m_data[idx].complex_packing();
    }
  }

  for (size_t idx = 0; idx < new_data.size(); ++idx) {
    new_data[idx].set_plaintext(new_plaintexts[idx]);
  }

  m_data = std::move(new_data);
  m_packed = true;
  m_packed_shape = HETensor::pack_shape(get_shape());
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

uint64_t HETensor::batch_size(const Shape& shape, bool packed) {
  if (packed && !shape.empty()) {
    return shape[0];
  }
  return 1;
}

size_t HETensor::get_batched_element_count() const {
  if (get_batch_size() == 0) {
    NGRAPH_CHECK(get_element_count() == 0,
                 "Non-zero element count with batch size 0");
    return 0;
  }
  return get_element_count() / get_batch_size();
}

bool HETensor::any_encrypted_data() const {
  return std::any_of(m_data.begin(), m_data.end(), [](const HEType& he_type) {
    return he_type.is_ciphertext();
  });
}

void HETensor::check_io_bounds(size_t n) const {
  size_t bytes_per_element = n;
  if (get_batch_size() == 0) {
    NGRAPH_CHECK(n == 0, "I/O access past end of tensor");
  } else {
    bytes_per_element /= get_batch_size();
  }

  const element::Type& element_type = get_tensor_layout()->get_element_type();
  size_t type_byte_size = element_type.size();

  // Memory must be byte-aligned to type_byte_size
  if (bytes_per_element % type_byte_size != 0) {
    throw ngraph_error("n must be divisible by type_byte_size.");
  }
  // Check out-of-range
  if (bytes_per_element / type_byte_size > get_element_count()) {
    throw std::out_of_range("I/O access past end of tensor");
  }
}

void HETensor::write(const void* p, size_t n) {
  check_io_bounds(n);

  const element::Type& element_type = get_tensor_layout()->get_element_type();
  size_t type_byte_size = element_type.size();

  size_t num_elements_to_write = n / element_type.size();
  if (get_batch_size() != 0) {
    num_elements_to_write /= get_batch_size();
  }

#pragma omp parallel for
  // NOLINTNEXTLINE
  for (size_t i = 0; i < num_elements_to_write; ++i) {
    HEPlaintext plain(get_batch_size());
    for (size_t j = 0; j < get_batch_size(); ++j) {
      const auto* src = static_cast<const void*>(
          static_cast<const char*>(p) +
          type_byte_size * (i + j * num_elements_to_write));
      plain[j] = type_to_double(src, element_type);
    }

    if (m_data[i].is_plaintext()) {
      m_data[i].set_plaintext(plain);
    } else {
      NGRAPH_CHECK(m_data[i].is_ciphertext(),
                   "Cannot write into tensor of unspecified type");
      auto cipher = HESealBackend::create_empty_ciphertext();

      encrypt(cipher, plain, m_context->first_parms_id(), element_type,
              m_encryption_params.scale(), m_ckks_encoder, m_encryptor,
              m_data[i].complex_packing());
      m_data[i].set_ciphertext(cipher);
    }
  }
  m_write_count += num_elements_to_write;
}

void HETensor::read(void* p, size_t n) const {
  check_io_bounds(n);
  const element::Type& element_type = get_tensor_layout()->get_element_type();
  size_t type_byte_size = element_type.size();
  size_t num_elements_to_read = n / (type_byte_size * get_batch_size());

  auto copy_batch_values_to_src = [&](size_t element_idx, void* copy_target,
                                      const void* type_values_src) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
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
  // NOLINTNEXTLINE
  for (size_t i = 0; i < num_elements_to_read; ++i) {
    HEPlaintext plain;
    if (m_data[i].is_ciphertext()) {
      decrypt(plain, *m_data[i].get_ciphertext(), m_data[i].complex_packing(),
              m_decryptor, m_ckks_encoder, m_context, m_data[i].batch_size());
    } else {
      plain = m_data[i].get_plaintext();
    }

    void* dst = ngraph_malloc(type_byte_size * get_batch_size());
    plain.resize(get_batch_size());
    plain.write(dst, element_type);

    copy_batch_values_to_src(i, p, dst);

    ngraph_free(dst);
  }
}

void HETensor::write_to_protos(std::vector<pb::HETensor>& proto_tensors) const {
  // Populate attributes of tensor to estimate byte size
  proto_tensors.resize(1);
  proto_tensors[0].set_name(get_name());
  std::vector<uint64_t> int_shape{get_shape()};
  *proto_tensors[0].mutable_shape() = {int_shape.begin(), int_shape.end()};
  proto_tensors[0].set_type(type_to_pb_type(get_element_type()));
  proto_tensors[0].set_packed(m_packed);
  proto_tensors[0].set_offset(0);

  NGRAPH_HE_LOG(5) << "Writing tensor shape " << get_shape();

  if (!m_data.empty()) {
    pb::HEType tmp_type;
    m_data[0].save(tmp_type);

    size_t he_type_size = tmp_type.ByteSize();
    size_t max_num_data_per_tensor =
        std::floor(std::numeric_limits<int32_t>::max() /
                   static_cast<float>(he_type_size)) -
        2;

    size_t num_tensors = m_data.size() / max_num_data_per_tensor;
    if (m_data.size() % max_num_data_per_tensor != 0) {
      num_tensors++;
    }
    proto_tensors.resize(num_tensors);

    size_t offset = 0;

    for (size_t tensor_idx = 0; tensor_idx < num_tensors; ++tensor_idx) {
      proto_tensors[tensor_idx].set_name(get_name());
      *proto_tensors[tensor_idx].mutable_shape() = {int_shape.begin(),
                                                    int_shape.end()};
      proto_tensors[tensor_idx].set_type(type_to_pb_type(get_element_type()));
      proto_tensors[tensor_idx].set_packed(m_packed);
      proto_tensors[tensor_idx].set_offset(offset);

      auto* mutable_data = proto_tensors[tensor_idx].mutable_data();
      size_t num_data_in_tensor = max_num_data_per_tensor;
      if (tensor_idx == num_tensors - 1) {
        num_data_in_tensor =
            m_data.size() - tensor_idx * max_num_data_per_tensor;
      }
      for (size_t data_idx = 0; data_idx < num_data_in_tensor; ++data_idx) {
        mutable_data->Add();
      }

#pragma omp parallel for
      // NOLINTNEXTLINE
      for (size_t data_idx = 0; data_idx < num_data_in_tensor; ++data_idx) {
        size_t data_offset = offset + data_idx;
        m_data[data_offset].save(*mutable_data->Mutable(data_idx));
      }
      offset += num_data_in_tensor;
    }
  }
}

std::shared_ptr<HETensor> HETensor::load_from_proto_tensors(
    const std::vector<pb::HETensor>& proto_tensors,
    seal::CKKSEncoder& ckks_encoder,
    const std::shared_ptr<seal::SEALContext>& context,
    const seal::Encryptor& encryptor, seal::Decryptor& decryptor,
    const HESealEncryptionParameters& encryption_params) {
  NGRAPH_CHECK(proto_tensors.size() == 1,
               "Load from protos only supports 1 proto");

  const auto& proto_tensor = proto_tensors[0];
  const auto& proto_name = proto_tensor.name();
  const auto& proto_shape = proto_tensor.shape();
  const auto& element_type = pb_type_to_type(proto_tensor.type());
  const auto& proto_packed = proto_tensor.packed();
  size_t result_count = proto_tensor.data_size();
  Shape shape{proto_shape.begin(), proto_shape.end()};

  auto he_tensor = std::make_shared<HETensor>(
      element_type, shape, proto_packed, encryption_params.complex_packing(),
      false, ckks_encoder, context, encryptor, decryptor, encryption_params,
      proto_name);

#pragma omp parallel for
  // NOLINTNEXTLINE
  for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
    const auto& loaded = HEType::load(proto_tensor.data(result_idx), context);
    he_tensor->data(result_idx) = loaded;
  }
  he_tensor->m_write_count += result_count;

  return he_tensor;
}

void HETensor::load_from_proto_tensor(
    std::shared_ptr<HETensor>& he_tensor, const pb::HETensor& proto_tensor,
    const std::shared_ptr<seal::SEALContext>& context) {
  const auto& proto_name = proto_tensor.name();
  const auto& proto_packed = proto_tensor.packed();
  const auto& proto_shape = proto_tensor.shape();
  const auto& proto_offset = proto_tensor.offset();
  size_t result_count = proto_tensor.data_size();
  Shape shape{proto_shape.begin(), proto_shape.end()};

  NGRAPH_CHECK(he_tensor != nullptr, "HETensor is empty");
  NGRAPH_CHECK(he_tensor->get_shape() == shape, "HETensor has wrong shape ",
               he_tensor->get_shape(), ", expected ", shape);
  NGRAPH_CHECK(he_tensor->get_name() == proto_name, "HETensor has wrong name ",
               he_tensor->get_name(), ", expected ", proto_name);
  NGRAPH_CHECK(he_tensor->is_packed() == proto_packed,
               "HETensor has wrong packing ", he_tensor->is_packed(),
               ", expected ", proto_packed);

#pragma omp parallel for
  // NOLINTNEXTLINE
  for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
    const auto& loaded = HEType::load(proto_tensor.data(result_idx), context);
    he_tensor->data(proto_offset + result_idx) = loaded;
  }
  he_tensor->m_write_count += result_count;
}

}  // namespace ngraph::runtime::he
