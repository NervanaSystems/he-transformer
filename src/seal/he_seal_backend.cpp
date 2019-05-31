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

#include <limits>
#include <memory>

#include "he_seal_cipher_tensor.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_executable.hpp"
#include "seal/seal.h"

extern "C" const char* get_ngraph_version_string() {
  return "DUMMY_VERSION";  // TODO: move to CMakeList
}

extern "C" ngraph::runtime::Backend* new_backend(
    const char* configuration_chars) {
  std::string configuration_string = std::string(configuration_chars);

  NGRAPH_CHECK(configuration_string == "HE_SEAL_CKKS",
               "Invalid configuration string ", configuration_string);
  return new ngraph::he::HESealBackend();
}

extern "C" void delete_backend(ngraph::runtime::Backend* backend) {
  delete backend;
}

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HESealBackend::create_tensor(const element::Type& element_type,
                                         const Shape& shape) {
  if (batch_data()) {
    return create_batched_plain_tensor(element_type, shape);
  } else {
    return create_plain_tensor(element_type, shape);
  }
}

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HESealBackend::create_cipher_tensor(
    const element::Type& element_type, const Shape& shape,
    const bool batched) const {
  auto rc = std::make_shared<ngraph::he::HESealCipherTensor>(
      element_type, shape, this, create_empty_ciphertext(), batched);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(rc);
}

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HESealBackend::create_valued_cipher_tensor(
    float value, const element::Type& element_type, const Shape& shape) const {
  auto tensor = std::static_pointer_cast<HESealCipherTensor>(
      create_cipher_tensor(element_type, shape));
  std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>&
      cipher_texts = tensor->get_elements();
#pragma omp parallel for
  for (size_t i = 0; i < cipher_texts.size(); ++i) {
    cipher_texts[i] = create_valued_ciphertext(value, element_type);
  }
  return tensor;
}

std::shared_ptr<ngraph::runtime::Executable> ngraph::he::HESealBackend::compile(
    std::shared_ptr<Function> function, bool enable_performance_collection) {
  return std::make_shared<HESealExecutable>(
      function, enable_performance_collection, this, m_encrypt_data,
      m_encrypt_model, m_batch_data);
}

std::shared_ptr<ngraph::he::SealCiphertextWrapper>
ngraph::he::HESealBackend::create_valued_ciphertext(
    float value, const element::Type& element_type, size_t batch_size) const {
  NGRAPH_CHECK(element_type == element::f32, "element type ", element_type,
               "unsupported");
  if (batch_size != 1) {
    throw ngraph_error(
        "HESealBackend::create_valued_ciphertext only supports batch size 1");
  }
  auto plaintext = create_valued_plaintext({value}, m_complex_packing);
  auto ciphertext = create_empty_ciphertext();

  encrypt(ciphertext, *plaintext);
  return ciphertext;
}

std::shared_ptr<ngraph::he::SealCiphertextWrapper>
ngraph::he::HESealBackend::create_empty_ciphertext() const {
  return std::make_shared<ngraph::he::SealCiphertextWrapper>();
}

std::shared_ptr<ngraph::he::SealCiphertextWrapper>
ngraph::he::HESealBackend::create_empty_ciphertext(
    seal::parms_id_type parms_id) const {
  return std::make_shared<ngraph::he::SealCiphertextWrapper>(
      seal::Ciphertext(m_context, parms_id));
}

std::shared_ptr<ngraph::he::SealCiphertextWrapper>
ngraph::he::HESealBackend::create_empty_ciphertext(
    const seal::MemoryPoolHandle& pool) const {
  return std::make_shared<ngraph::he::SealCiphertextWrapper>(pool);
}

void ngraph::he::HESealBackend::encrypt(
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& output,
    const ngraph::he::HEPlaintext& input) const {
  auto seal_wrapper = make_seal_plaintext_wrapper(input.complex_packing());

  encode(*seal_wrapper, input);
  // No need to encrypt zero
  if (input.is_single_value() && input.get_values()[0] == 0) {
    output->set_zero(true);
  } else {
    m_encryptor->encrypt(seal_wrapper->plaintext(), output->ciphertext());
  }
  output->set_complex_packing(input.complex_packing());
  NGRAPH_CHECK(output->complex_packing() == input.complex_packing());
}

void ngraph::he::HESealBackend::decrypt(
    ngraph::he::HEPlaintext& output,
    const std::shared_ptr<ngraph::he::SealCiphertextWrapper>& input) const {
  if (input->is_zero()) {
    // TOOD: refine?
    const size_t slots =
        m_context->context_data()->parms().poly_modulus_degree() / 2;
    output.set_values(std::vector<float>(slots, 0));
  } else {
    auto plaintext_wrapper =
        make_seal_plaintext_wrapper(input->complex_packing());
    m_decryptor->decrypt(input->ciphertext(), plaintext_wrapper->plaintext());
    decode(output, *plaintext_wrapper);
  }
  output.set_complex_packing(input->complex_packing());
  NGRAPH_CHECK(output.complex_packing() == input->complex_packing());
}
