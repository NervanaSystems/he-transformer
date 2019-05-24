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

#include "seal/bfv/he_seal_bfv_backend.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"

using namespace ngraph;
using namespace std;

extern "C" const char* get_ngraph_version_string() {
  return "DUMMY_VERSION";  // TODO: move to CMakeList
}

extern "C" runtime::Backend* new_backend(const char* configuration_chars) {
  string configuration_string = string(configuration_chars);

  if (configuration_string == "HE_SEAL_BFV") {
    return new runtime::he::he_seal::HESealBFVBackend();
  } else if (configuration_string == "HE_SEAL_CKKS") {
    return new runtime::he::he_seal::HESealCKKSBackend();
  } else {
    throw ngraph_error("Invalid configuration string \"" +
                       configuration_string + "\" in new_backend");
  }
}

extern "C" void delete_backend(runtime::Backend* backend) { delete backend; }

shared_ptr<runtime::he::HECiphertext>
runtime::he::he_seal::HESealBackend::create_empty_ciphertext() const {
  return make_shared<runtime::he::he_seal::SealCiphertextWrapper>();
}

shared_ptr<runtime::he::HECiphertext>
runtime::he::he_seal::HESealBackend::create_empty_ciphertext(
    seal::parms_id_type parms_id) const {
  return make_shared<runtime::he::he_seal::SealCiphertextWrapper>(
      seal::Ciphertext(m_context, parms_id));
}

shared_ptr<runtime::he::HECiphertext>
runtime::he::he_seal::HESealBackend::create_empty_ciphertext(
    const seal::MemoryPoolHandle& pool) const {
  return make_shared<runtime::he::he_seal::SealCiphertextWrapper>(pool);
}

shared_ptr<runtime::he::HEPlaintext>
runtime::he::he_seal::HESealBackend::create_empty_plaintext() const {
  return make_shared<runtime::he::he_seal::SealPlaintextWrapper>();
}

shared_ptr<runtime::he::HEPlaintext>
runtime::he::he_seal::HESealBackend::create_empty_plaintext(
    const seal::MemoryPoolHandle& pool) const {
  return make_shared<runtime::he::he_seal::SealPlaintextWrapper>(pool);
}

void runtime::he::he_seal::HESealBackend::encrypt(
    shared_ptr<runtime::he::HECiphertext>& output,
    shared_ptr<runtime::he::HEPlaintext>& input) const {
  auto seal_output = runtime::he::he_seal::cast_to_seal_hetext(output);
  auto seal_input = runtime::he::he_seal::cast_to_seal_hetext(input);

  encode(seal_input, input->complex_packing());
  // No need to encrypt zero
  if (input->is_single_value() && input->get_values()[0] == 0) {
    seal_output->set_zero(true);
  } else {
    m_encryptor->encrypt(seal_input->get_plaintext(),
                         seal_output->m_ciphertext);
  }
  output->set_complex_packing(input->complex_packing());
  NGRAPH_ASSERT(output->complex_packing() == input->complex_packing());
}

void runtime::he::he_seal::HESealBackend::decrypt(
    shared_ptr<runtime::he::HEPlaintext>& output,
    const shared_ptr<runtime::he::HECiphertext>& input) const {
  auto seal_output = runtime::he::he_seal::cast_to_seal_hetext(output);
  auto seal_input = runtime::he::he_seal::cast_to_seal_hetext(input);

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
  NGRAPH_ASSERT(output->complex_packing() == input->complex_packing());
}
