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
  return "v0.9.1";  // TODO: move to CMakeLists
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

void runtime::he::he_seal::HESealBackend::assert_valid_seal_parameter(
    const shared_ptr<runtime::he::he_seal::HESealParameter> sp) const {
  if (sp->m_scheme_name != "HE_SEAL_BFV" &&
      sp->m_scheme_name != "HE_SEAL_CKKS") {
    throw ngraph_error("Invalid scheme name");
  }
  static unordered_set<uint64_t> valid_poly_modulus{1024, 2048,  4096,
                                                    8192, 16384, 32768};
  if (valid_poly_modulus.count(sp->m_poly_modulus_degree) == 0) {
    throw ngraph_error(
        "m_poly_modulus must be 1024, 2048, 4096, 8192, 16384, 32768");
  }

  if (sp->m_security_level != 128 && sp->m_security_level != 192) {
    throw ngraph_error("sp.security_level must be 128, 192");
  }

  if (sp->m_evaluation_decomposition_bit_count > 60 ||
      sp->m_evaluation_decomposition_bit_count < 1) {
    throw ngraph_error(
        "sp.m_evaluation_decomposition_bit_count must be between 1 and 60");
  }
}

shared_ptr<runtime::he::HECiphertext>
runtime::he::he_seal::HESealBackend::create_empty_ciphertext() const {
  return make_shared<runtime::he::he_seal::SealCiphertextWrapper>();
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
    const runtime::he::HEPlaintext& input) const {
  auto seal_output =
      static_pointer_cast<runtime::he::he_seal::SealCiphertextWrapper>(output);
  auto seal_input =
      static_cast<const runtime::he::he_seal::SealPlaintextWrapper&>(input);
  m_encryptor->encrypt(seal_input.m_plaintext, seal_output->m_ciphertext);
}

void runtime::he::he_seal::HESealBackend::decrypt(
    shared_ptr<runtime::he::HEPlaintext>& output,
    const runtime::he::HECiphertext& input) const {
  auto seal_output =
      dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(output);
  auto seal_input =
      static_cast<const runtime::he::he_seal::SealCiphertextWrapper&>(input);
  m_decryptor->decrypt(seal_input.m_ciphertext, seal_output->m_plaintext);
}

const shared_ptr<const runtime::he::HEPlaintext>
runtime::he::he_seal::HESealBackend::get_valued_plaintext(double value) const {
  NGRAPH_ASSERT(m_plaintext_map.find(value) != m_plaintext_map.end())
      << "Plaintext value " << value << " not found";
  return m_plaintext_map.at(value);
}