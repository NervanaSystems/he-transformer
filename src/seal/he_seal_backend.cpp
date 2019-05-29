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

#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"

extern "C" const char* get_ngraph_version_string() {
  return "DUMMY_VERSION";  // TODO: move to CMakeList
}

extern "C" ngraph::runtime::Backend* new_backend(
    const char* configuration_chars) {
  std::string configuration_string = std::string(configuration_chars);

  NGRAPH_CHECK(configuration_string == "HE_SEAL_CKKS",
               "Invalid configuration string ", configuration_string);
  return new ngraph::he::HESealCKKSBackend();
}

extern "C" void delete_backend(ngraph::runtime::Backend* backend) {
  delete backend;
}

std::shared_ptr<ngraph::he::HECiphertext>
ngraph::he::HESealBackend::create_empty_ciphertext() const {
  return std::make_shared<ngraph::he::SealCiphertextWrapper>();
}

std::shared_ptr<ngraph::he::HECiphertext>
ngraph::he::HESealBackend::create_empty_ciphertext(
    seal::parms_id_type parms_id) const {
  return std::make_shared<ngraph::he::SealCiphertextWrapper>(
      seal::Ciphertext(m_context, parms_id));
}

std::shared_ptr<ngraph::he::HECiphertext>
ngraph::he::HESealBackend::create_empty_ciphertext(
    const seal::MemoryPoolHandle& pool) const {
  return std::make_shared<ngraph::he::SealCiphertextWrapper>(pool);
}

std::shared_ptr<ngraph::he::HEPlaintext>
ngraph::he::HESealBackend::create_empty_plaintext() const {
  return std::make_shared<ngraph::he::SealPlaintextWrapper>();
}

std::shared_ptr<ngraph::he::HEPlaintext>
ngraph::he::HESealBackend::create_empty_plaintext(
    const seal::MemoryPoolHandle& pool) const {
  return std::make_shared<ngraph::he::SealPlaintextWrapper>(pool);
}

void ngraph::he::HESealBackend::encrypt(
    std::shared_ptr<ngraph::he::HECiphertext>& output,
    std::shared_ptr<ngraph::he::HEPlaintext>& input) const {
  auto seal_output = ngraph::he::cast_to_seal_hetext(output);
  auto seal_input = ngraph::he::cast_to_seal_hetext(input);

  encode(seal_input, input->complex_packing());
  // No need to encrypt zero
  if (input->is_single_value() && input->get_values()[0] == 0) {
    seal_output->set_zero(true);
  } else {
    m_encryptor->encrypt(seal_input->get_plaintext(),
                         seal_output->m_ciphertext);
  }
  output->set_complex_packing(input->complex_packing());
  NGRAPH_CHECK(output->complex_packing() == input->complex_packing());
}

void ngraph::he::HESealBackend::decrypt(
    std::shared_ptr<ngraph::he::HEPlaintext>& output,
    const std::shared_ptr<ngraph::he::HECiphertext>& input) const {
  auto seal_output = ngraph::he::cast_to_seal_hetext(output);
  auto seal_input = ngraph::he::cast_to_seal_hetext(input);

  if (input->is_zero()) {
    const size_t slots =
        m_context->context_data()->parms().poly_modulus_degree() / 2;
    output->set_values(std::vector<float>(slots, 0));

    // TODO: placeholder until we figure out how to decode/encode plaintexts
    // properly
    encode(seal_output, input->complex_packing());
    output->set_encoded(true);
  } else {
    m_decryptor->decrypt(seal_input->m_ciphertext,
                         seal_output->get_plaintext());
  }
  output->set_complex_packing(input->complex_packing());
  NGRAPH_CHECK(output->complex_packing() == input->complex_packing());
}
