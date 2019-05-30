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

ngraph::he::HESealCKKSBackend::HESealCKKSBackend()
    : ngraph::he::HESealCKKSBackend(
          ngraph::he::parse_config_or_use_default("HE_SEAL_CKKS")) {}

ngraph::he::HESealCKKSBackend::HESealCKKSBackend(
    const std::shared_ptr<ngraph::he::HEEncryptionParameters>& sp) {
  auto he_seal_encryption_parms =
      std::static_pointer_cast<ngraph::he::HESealEncryptionParameters>(sp);

  NGRAPH_CHECK(he_seal_encryption_parms != nullptr,
               "HE_SEAL_CKKS backend passed invalid encryption parameters");
  m_context = seal::SEALContext::Create(
      *(he_seal_encryption_parms->seal_encryption_parameters()));

  m_encryption_params = sp;

  print_seal_context(*m_context);

  auto context_data = m_context->context_data();

  // Keygen, encryptor and decryptor
  m_keygen = std::make_shared<seal::KeyGenerator>(m_context);
  m_relin_keys = std::make_shared<seal::RelinKeys>(
      m_keygen->relin_keys(sp->evaluation_decomposition_bit_count()));
  m_public_key = std::make_shared<seal::PublicKey>(m_keygen->public_key());
  m_secret_key = std::make_shared<seal::SecretKey>(m_keygen->secret_key());
  m_encryptor = std::make_shared<seal::Encryptor>(m_context, *m_public_key);
  m_decryptor = std::make_shared<seal::Decryptor>(m_context, *m_secret_key);

  // Evaluator
  m_evaluator = std::make_shared<seal::Evaluator>(m_context);

  m_scale =
      static_cast<double>(context_data->parms().coeff_modulus().back().value());

  // Encoder
  m_ckks_encoder = std::make_shared<seal::CKKSEncoder>(m_context);
}

extern "C" ngraph::runtime::Backend* new_ckks_backend(
    const char* configuration_string) {
  return new ngraph::he::HESealCKKSBackend();
}

std::shared_ptr<seal::SEALContext>
ngraph::he::HESealCKKSBackend::make_seal_context(
    const std::shared_ptr<ngraph::he::HEEncryptionParameters> sp) {
  throw ngraph_error("make SEAL context unimplemented");
}

namespace {
static class HESealCKKSStaticInit {
 public:
  HESealCKKSStaticInit() {
    ngraph::runtime::BackendManager::register_backend("HE_SEAL_CKKS",
                                                      new_ckks_backend);
  }
  ~HESealCKKSStaticInit() {}
} s_he_seal_ckks_static_init;
}  // namespace

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HESealCKKSBackend::create_batched_cipher_tensor(
    const element::Type& type, const Shape& shape) {
  NGRAPH_INFO << "Creating batched cipher tensor with shape " << join(shape);
  auto rc = std::make_shared<ngraph::he::HECipherTensor>(
      type, shape, this, create_empty_ciphertext(), true);
  set_batch_data(true);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(rc);
}

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HESealCKKSBackend::create_batched_plain_tensor(
    const element::Type& type, const Shape& shape) {
  NGRAPH_INFO << "Creating batched plain tensor with shape " << join(shape);
  auto rc = std::make_shared<ngraph::he::HEPlainTensor>(
      type, shape, this, create_empty_plaintext(), true);
  set_batch_data(true);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(rc);
}

void ngraph::he::HESealCKKSBackend::encode(
    const ngraph::he::HEPlaintext& plaintext,
    ngraph::he::SealPlaintextWrapper& destination, seal::parms_id_type parms_id,
    double scale) const {
  throw ngraph_error("Uimplemented");
}

void ngraph::he::HESealCKKSBackend::encode(
    const ngraph::he::HEPlaintext& plaintext,
    ngraph::he::SealPlaintextWrapper& destination) const {
  throw ngraph_error("Uimplemented");
}

void ngraph::he::HESealCKKSBackend::encode(ngraph::he::HEPlaintext& output,
                                           const void* input,
                                           const element::Type& type,
                                           bool complex, size_t count) const {
  throw ngraph_error("Uimplemented");
}
/*
void ngraph::he::HESealCKKSBackend::encode(
    ngraph::he::HEPlaintext& plaintext,
    seal::parms_id_type parms_id, double scale, bool complex) const {
  NGRAPH_INFO << "Encode";
  throw ngraph_error("Uimplemented");
   std::lock_guard<std::mutex> encode_lock(plaintext->get_encode_mutex());
  if (plaintext->is_encoded()) {
    return;
  }

   std::vector<double> double_vals(plaintext->get_values().begin(),
                                   plaintext->get_values().end());

   const size_t slots =
       m_context->context_data()->parms().poly_modulus_degree() / 2;
   if (complex) {
     std::vector<std::complex<double>> complex_vals;
     if (double_vals.size() == 1) {
       std::complex<double> val(double_vals[0], double_vals[0]);
       complex_vals = std::vector<std::complex<double>>(slots, val);
     } else {
       real_vec_to_complex_vec(complex_vals, double_vals);
     }
     NGRAPH_CHECK(complex_vals.size() <= slots, "Cannot encode ",
                  complex_vals.size(), " elements, maximum size is ", slots);
     m_ckks_encoder->encode(complex_vals, parms_id, scale,
                            plaintext->get_plaintext());
   } else {
     // TODO: why different cases?
     if (double_vals.size() == 1) {
       m_ckks_encoder->encode(double_vals[0], parms_id, scale,
                              plaintext->get_plaintext());
     } else {
       NGRAPH_CHECK(double_vals.size() <= slots, "Cannot encode ",
                    double_vals.size(), " elements, maximum size is ", slots);
       m_ckks_encoder->encode(double_vals, parms_id, scale,
                              plaintext->get_plaintext());
     }
   }
   plaintext->set_complex_packing(complex);
   plaintext->set_encoded(true);
}
*/

/*
void ngraph::he::HESealCKKSBackend::encode(
    ngraph::he::HEPlaintext& output, const void* input,
    const element::Type& type, bool complex, size_t count) const {
  NGRAPH_INFO << "Encode";
  throw ngraph_error("Unimplemented");
  auto seal_plaintext_wrapper = cast_to_seal_hetext(output);

   NGRAPH_CHECK(type == element::f32,
                "CKKS encode supports only float encoding, received type ",
                type);

   std::vector<float> values{(float*)input, (float*)input + count};
   seal_plaintext_wrapper->set_values(values);

   encode(seal_plaintext_wrapper, complex);
}
*/

void ngraph::he::HESealCKKSBackend::decode(void* output,
                                           const ngraph::he::HEPlaintext& input,
                                           const element::Type& type,
                                           size_t count) const {
  NGRAPH_CHECK(count != 0, "Decode called on 0 elements");
  NGRAPH_CHECK(type == element::f32,
               "CKKS encode supports only float encoding, received type ",
               type);
  std::vector<float> xs_float = input.get_values();

  NGRAPH_CHECK(xs_float.size() >= count);
  std::memcpy(output, &xs_float[0], type.size() * count);
}

void ngraph::he::HESealCKKSBackend::decode(
    ngraph::he::HEPlaintext& output, const ngraph::he::SealPlaintextWrapper&,
    input) const {
  std::vector<double> real_vals;
  if (input.complex_packing) {
    std::vector<std::complex<double>> complex_vals;
    m_ckks_encoder->decode(input.m_plaintext, complex_vals);
    complex_vec_to_real_vec(real_vals, complex_vals);
  } else {
    m_ckks_encoder->decode(input.m_plaintext, real_vals);
  }
  std::vector<float> float_vals{real_vals.begin(), real_vals.end()};
  output.set_values(float_vals);
  output.set_complex_packing(input.complex_packing);
}
