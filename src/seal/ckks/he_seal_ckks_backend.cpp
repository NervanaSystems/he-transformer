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

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>

#include "he_cipher_tensor.hpp"
#include "he_plain_tensor.hpp"
#include "he_tensor.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/backend_manager.hpp"

#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_encryption_parameters.hpp"
#include "seal/he_seal_util.hpp"
#include "seal/seal.h"
#include "tcp/tcp_message.hpp"

using namespace ngraph;
using namespace std;

runtime::he::he_seal::HESealCKKSBackend::HESealCKKSBackend()
    : runtime::he::he_seal::HESealCKKSBackend(
          runtime::he::he_seal::parse_config_or_use_default("HE_SEAL_CKKS")) {}

runtime::he::he_seal::HESealCKKSBackend::HESealCKKSBackend(
    const shared_ptr<runtime::he::HEEncryptionParameters>& sp) {
  auto he_seal_encryption_parms =
      static_pointer_cast<runtime::he::he_seal::HESealEncryptionParameters>(sp);

  NGRAPH_ASSERT(he_seal_encryption_parms != nullptr)
      << "HE_SEAL_CKKS backend passed invalid encryption parameters";
  m_context = seal::SEALContext::Create(
      *(he_seal_encryption_parms->seal_encryption_parameters()));

  print_seal_context(*m_context);

  auto context_data = m_context->context_data();

  // Keygen, encryptor and decryptor
  m_keygen = make_shared<seal::KeyGenerator>(m_context);
  m_relin_keys = make_shared<seal::RelinKeys>(
      m_keygen->relin_keys(sp->evaluation_decomposition_bit_count()));
  m_public_key = make_shared<seal::PublicKey>(m_keygen->public_key());
  m_secret_key = make_shared<seal::SecretKey>(m_keygen->secret_key());
  m_encryptor = make_shared<seal::Encryptor>(m_context, *m_public_key);
  m_decryptor = make_shared<seal::Decryptor>(m_context, *m_secret_key);

  // Evaluator
  m_evaluator = make_shared<seal::Evaluator>(m_context);

  m_scale =
      static_cast<double>(context_data->parms().coeff_modulus().back().value());

  // Encoder
  m_ckks_encoder = make_shared<seal::CKKSEncoder>(m_context);

  // Plaintext constants
  shared_ptr<runtime::he::HEPlaintext> plaintext_neg1 =
      create_empty_plaintext();
  shared_ptr<runtime::he::HEPlaintext> plaintext_0 = create_empty_plaintext();
  shared_ptr<runtime::he::HEPlaintext> plaintext_1 = create_empty_plaintext();

  m_ckks_encoder->encode(
      -1, m_scale,
      dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(
          plaintext_neg1)
          ->m_plaintext);

  m_ckks_encoder->encode(
      0, m_scale,
      dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(
          plaintext_0)
          ->m_plaintext);

  m_ckks_encoder->encode(
      1, m_scale,
      dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(
          plaintext_1)
          ->m_plaintext);

  m_plaintext_map[-1] = plaintext_neg1;
  m_plaintext_map[0] = plaintext_0;
  m_plaintext_map[1] = plaintext_1;
}

extern "C" runtime::Backend* new_ckks_backend(
    const char* configuration_string) {
  return new runtime::he::he_seal::HESealCKKSBackend();
}

shared_ptr<seal::SEALContext>
runtime::he::he_seal::HESealCKKSBackend::make_seal_context(
    const shared_ptr<runtime::he::HEEncryptionParameters> sp) {
  throw ngraph_error("make SEAL context unimplemented");
}

namespace {
static class HESealCKKSStaticInit {
 public:
  HESealCKKSStaticInit() {
    runtime::BackendManager::register_backend("HE_SEAL_CKKS", new_ckks_backend);
  }
  ~HESealCKKSStaticInit() {}
} s_he_seal_ckks_static_init;
}  // namespace

shared_ptr<runtime::Tensor>
runtime::he::he_seal::HESealCKKSBackend::create_batched_cipher_tensor(
    const element::Type& element_type, const Shape& shape) {
  NGRAPH_INFO << "Creating batched cipher tensor with shape " << join(shape);
  auto rc = make_shared<runtime::he::HECipherTensor>(
      element_type, shape, this, create_empty_ciphertext(), true);
  return static_pointer_cast<runtime::Tensor>(rc);
}

shared_ptr<runtime::Tensor>
runtime::he::he_seal::HESealCKKSBackend::create_batched_plain_tensor(
    const element::Type& element_type, const Shape& shape) {
  NGRAPH_INFO << "Creating batched plain tensor with shape " << join(shape);
  auto rc = make_shared<runtime::he::HEPlainTensor>(
      element_type, shape, this, create_empty_plaintext(), true);
  return static_pointer_cast<runtime::Tensor>(rc);
}

void runtime::he::he_seal::HESealCKKSBackend::encode(
    shared_ptr<runtime::he::HEPlaintext>& output, const void* input,
    const element::Type& element_type, size_t count) const {
  const string type_name = element_type.c_type_string();
  if (type_name == "float") {
    if (count == 1) {
      double value = (double)(*(float*)input);
      if (m_plaintext_map.find(value) != m_plaintext_map.end()) {
        auto plain_value = static_pointer_cast<
            const runtime::he::he_seal::SealPlaintextWrapper>(
            get_valued_plaintext(value));
        output = make_shared<runtime::he::he_seal::SealPlaintextWrapper>(
            *plain_value);
      } else {
        m_ckks_encoder->encode(
            value, m_scale,
            dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(
                output)
                ->m_plaintext);
      }
    } else {
      vector<float> values{(float*)input, (float*)input + count};
      vector<double> double_values(values.begin(), values.end());

      m_ckks_encoder->encode(
          double_values, m_scale,
          dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(
              output)
              ->m_plaintext);
    }
  } else {
    NGRAPH_INFO << "Unsupported element type in encode " << type_name;
    throw ngraph_error("Unsupported element type " + type_name);
  }
}

void runtime::he::he_seal::HESealCKKSBackend::decode(
    void* output, const runtime::he::HEPlaintext* input,
    const element::Type& element_type, size_t count) const {
  const string type_name = element_type.c_type_string();

  if (count == 0) {
    throw ngraph_error("Decode called on 0 elements");
  }

  if (type_name == "float") {
    auto seal_input = dynamic_cast<const SealPlaintextWrapper*>(input);
    if (!seal_input) {
      throw ngraph_error(
          "HESealCKKSBackend::decode input is not seal plaintext");
    }
    vector<double> xs;
    m_ckks_encoder->decode(seal_input->m_plaintext, xs);
    vector<float> xs_float(xs.begin(), xs.end());

    memcpy(output, &xs_float[0], element_type.size() * count);
  } else {
    throw ngraph_error("Unsupported element type " + type_name);
  }
}
