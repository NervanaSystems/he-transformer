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
#include <complex>
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

  m_encryption_params = sp;

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
    const element::Type& type, const Shape& shape) {
  NGRAPH_INFO << "Creating batched cipher tensor with shape " << join(shape);
  auto rc = make_shared<runtime::he::HECipherTensor>(
      type, shape, this, create_empty_ciphertext(), true);
  set_batch_data(true);
  return static_pointer_cast<runtime::Tensor>(rc);
}

shared_ptr<runtime::Tensor>
runtime::he::he_seal::HESealCKKSBackend::create_batched_plain_tensor(
    const element::Type& type, const Shape& shape) {
  NGRAPH_INFO << "Creating batched plain tensor with shape " << join(shape);
  auto rc = make_shared<runtime::he::HEPlainTensor>(
      type, shape, this, create_empty_plaintext(), true);
  set_batch_data(true);
  return static_pointer_cast<runtime::Tensor>(rc);
}

void runtime::he::he_seal::HESealCKKSBackend::encode(
    std::vector<std::shared_ptr<runtime::he::he_seal::SealPlaintextWrapper>>&
        plaintexts,
    bool complex) const {
#pragma omp parallel for
  for (size_t i = 0; i < plaintexts.size(); ++i) {
    auto plaintext = plaintexts[i];
    encode(plaintext.get(), complex);
  }
}

void runtime::he::he_seal::HESealCKKSBackend::encode(
    runtime::he::he_seal::SealPlaintextWrapper* plaintext, bool complex) const {
  std::lock_guard<std::mutex> encode_lock(plaintext->get_encode_mutex());
  if (plaintext->is_encoded()) {
    return;
  }

  vector<double> double_vals(plaintext->get_values().begin(),
                             plaintext->get_values().end());

  const size_t slots =
      m_context->context_data()->parms().poly_modulus_degree() / 2;
  if (complex) {
    vector<std::complex<double>> complex_vals;
    if (double_vals.size() == 1) {
      std::complex<double> val(double_vals[0], double_vals[0]);
      complex_vals = std::vector<std::complex<double>>(slots, val);
    } else {
      real_vec_to_complex_vec(complex_vals, double_vals);
    }
    NGRAPH_ASSERT(complex_vals.size() <= slots)
        << "Cannot encode " << complex_vals.size()
        << " elements, maximum size is " << slots;
    m_ckks_encoder->encode(complex_vals, m_scale, plaintext->get_plaintext());
  } else {
    // TODO: why different cases?
    if (double_vals.size() == 1) {
      m_ckks_encoder->encode(double_vals[0], m_scale,
                             plaintext->get_plaintext());
    } else {
      NGRAPH_ASSERT(double_vals.size() <= slots)
          << "Cannot encode " << double_vals.size()
          << " elements, maximum size is " << slots;
      m_ckks_encoder->encode(double_vals, m_scale, plaintext->get_plaintext());
    }
  }
  plaintext->set_complex_packing(complex);
  plaintext->set_encoded(true);
}

void runtime::he::he_seal::HESealCKKSBackend::encode(
    shared_ptr<runtime::he::HEPlaintext>& output, const void* input,
    const element::Type& type, bool complex, size_t count) const {
  auto seal_plaintext_wrapper =
      dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(output);

  NGRAPH_ASSERT(seal_plaintext_wrapper != nullptr)
      << "HEPlaintext is not SealPlaintextWrapper";

  NGRAPH_ASSERT(type == element::f32)
      << "CKKS encode supports only float encoding, received type " << type;

  vector<float> values{(float*)input, (float*)input + count};
  seal_plaintext_wrapper->set_values(values);

  encode(seal_plaintext_wrapper.get(), complex);
}

void runtime::he::he_seal::HESealCKKSBackend::decode(
    void* output, runtime::he::HEPlaintext* input, const element::Type& type,
    size_t count) const {
  NGRAPH_ASSERT(count != 0) << "Decode called on 0 elements";
  NGRAPH_ASSERT(type == element::f32)
      << "CKKS encode supports only float encoding, received type " << type;
  decode(input);

  vector<float> xs_float = input->get_values();

  NGRAPH_ASSERT(xs_float.size() >= count);
  memcpy(output, &xs_float[0], type.size() * count);
}

void runtime::he::he_seal::HESealCKKSBackend::decode(
    runtime::he::HEPlaintext* input) const {
  auto seal_input = dynamic_cast<const SealPlaintextWrapper*>(input);
  NGRAPH_ASSERT(seal_input != nullptr)
      << "HESealCKKSBackend::decode input is not seal plaintext";

  vector<double> real_vals;
  if (input->complex_packing()) {
    vector<std::complex<double>> complex_vals;
    m_ckks_encoder->decode(seal_input->get_plaintext(), complex_vals);
    complex_vec_to_real_vec(real_vals, complex_vals);
  } else {
    m_ckks_encoder->decode(seal_input->get_plaintext(), real_vals);
  }
  vector<float> float_vals{real_vals.begin(), real_vals.end()};
  input->set_values(float_vals);
}