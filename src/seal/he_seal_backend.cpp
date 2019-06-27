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

#include "client_util.hpp"
#include "he_plain_tensor.hpp"
#include "he_seal_cipher_tensor.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_executable.hpp"
#include "seal/seal.h"
#include "seal/seal_util.hpp"

extern "C" const char* get_ngraph_version_string() {
  return "DUMMY_VERSION";  // TODO: move to CMakeList
}

// TODO: replace with new backend constructor once switching to new ngraph
extern "C" ngraph::runtime::Backend* new_backend(const char* config) {
  std::string configuration_string = std::string(config);

  NGRAPH_CHECK(configuration_string == "HE_SEAL",
               "Invalid configuration string ", configuration_string);
  return new ngraph::he::HESealBackend();
}

ngraph::he::HESealBackend::HESealBackend()
    : ngraph::he::HESealBackend(
          ngraph::he::parse_config_or_use_default("HE_SEAL")) {}

ngraph::he::HESealBackend::HESealBackend(
    const ngraph::he::HESealEncryptionParameters& parms)
    : m_encryption_params(parms) {
  seal::sec_level_type sec_level = seal::sec_level_type::none;
  if (parms.security_level() == 128) {
    sec_level = seal::sec_level_type::tc128;
  } else if (parms.security_level() == 192) {
    sec_level = seal::sec_level_type::tc192;
  } else if (parms.security_level() == 256) {
    sec_level = seal::sec_level_type::tc256;
  } else if (parms.security_level() == 0) {
    NGRAPH_WARN
        << "Parameter selection does not enforce minimum security level";
  } else {
    throw ngraph_error("Invalid security level");
  }

  m_context = seal::SEALContext::Create(parms.seal_encryption_parameters(),
                                        true, sec_level);

  print_seal_context(*m_context);

  auto context_data = m_context->key_context_data();

  // Keygen, encryptor and decryptor
  m_keygen = std::make_shared<seal::KeyGenerator>(m_context);
  m_relin_keys = std::make_shared<seal::RelinKeys>(m_keygen->relin_keys());
  m_public_key = std::make_shared<seal::PublicKey>(m_keygen->public_key());
  m_secret_key = std::make_shared<seal::SecretKey>(m_keygen->secret_key());
  m_encryptor = std::make_shared<seal::Encryptor>(m_context, *m_public_key);
  m_decryptor = std::make_shared<seal::Decryptor>(m_context, *m_secret_key);

  // Evaluator
  m_evaluator = std::make_shared<seal::Evaluator>(m_context);

  // TODO: pick smaller scale?
  auto coeff_moduli = context_data->parms().coeff_modulus();
  m_scale = ngraph::he::choose_scale(coeff_moduli);
  NGRAPH_INFO << "Scale " << m_scale;

  // Encoder
  m_ckks_encoder = std::make_shared<seal::CKKSEncoder>(m_context);

  // Set barrett ratio map
  for (const seal::SmallModulus& modulus : coeff_moduli) {
    const std::uint64_t modulus_value = modulus.value();
    if (modulus_value < (1UL << 31)) {
      std::uint64_t numerator[3]{0, 1};
      std::uint64_t quotient[3]{0, 0};
      seal::util::divide_uint128_uint64_inplace(numerator, modulus_value,
                                                quotient);
      std::uint64_t const_ratio = quotient[0];

      NGRAPH_CHECK(quotient[1] == 0, "Quotient[1] != 0 for modulus");
      m_barrett64_ratio_map[modulus_value] = const_ratio;
    }
  }

  NGRAPH_CHECK(!(m_encrypt_model && m_complex_packing),
               "NGRAPH_ENCRYPT_MODEL is incompatible with NGRAPH_COMPLEX_PACK");
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
ngraph::he::HESealBackend::create_plain_tensor(
    const element::Type& element_type, const Shape& shape,
    const bool batched) const {
  auto rc = std::make_shared<ngraph::he::HEPlainTensor>(element_type, shape,
                                                        *this, batched);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(rc);
}

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HESealBackend::create_cipher_tensor(
    const element::Type& element_type, const Shape& shape,
    const bool batched) const {
  auto rc = std::make_shared<ngraph::he::HESealCipherTensor>(
      element_type, shape, *this, batched);
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

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HESealBackend::create_batched_cipher_tensor(
    const element::Type& type, const Shape& shape) {
  auto rc = std::make_shared<ngraph::he::HESealCipherTensor>(type, shape, *this,
                                                             true);
  set_batch_data(true);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(rc);
}

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HESealBackend::create_batched_plain_tensor(
    const element::Type& type, const Shape& shape) {
  auto rc =
      std::make_shared<ngraph::he::HEPlainTensor>(type, shape, *this, true);
  set_batch_data(true);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(rc);
}

std::shared_ptr<ngraph::runtime::Executable> ngraph::he::HESealBackend::compile(
    std::shared_ptr<Function> function, bool enable_performance_collection) {
  return std::make_shared<HESealExecutable>(
      function, enable_performance_collection, *this, m_encrypt_data,
      m_encrypt_model, m_batch_data, m_complex_packing);
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
  auto plaintext = HEPlaintext({value});
  auto ciphertext = create_empty_ciphertext();

  encrypt(ciphertext, plaintext, complex_packing());
  return ciphertext;
}

void ngraph::he::HESealBackend::encrypt(
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& output,
    const ngraph::he::HEPlaintext& input, bool complex_packing) const {
  auto plaintext = SealPlaintextWrapper(complex_packing);

  NGRAPH_CHECK(input.num_values() > 0, "Input has no values in encrypt");

  encode(plaintext, input, complex_packing);
  // No need to encrypt single zero value.
  // TODO: encrypt 0 value!
  /* if (input.is_single_value() && input.values()[0] == 0) {
    NGRAPH_INFO << "Skipping encrypting 0 value";
    output->known_value() = true;
    output->value() = 0;

  } else { */
  m_encryptor->encrypt(plaintext.plaintext(), output->ciphertext());
  //}
  output->complex_packing() = complex_packing;
  output->known_value() = false;
}

void ngraph::he::HESealBackend::decrypt(
    ngraph::he::HEPlaintext& output,
    const ngraph::he::SealCiphertextWrapper& input) const {
  if (input.known_value()) {
    NGRAPH_INFO << "Decrypting known value " << input.value();
    const size_t slot_count = m_ckks_encoder->slot_count();
    output.values() = std::vector<float>(slot_count, input.value());
  } else {
    auto plaintext_wrapper = SealPlaintextWrapper(input.complex_packing());
    m_decryptor->decrypt(input.ciphertext(), plaintext_wrapper.plaintext());
    decode(output, plaintext_wrapper);
  }
}

void ngraph::he::HESealBackend::decode(void* output,
                                       const ngraph::he::HEPlaintext& input,
                                       const element::Type& type,
                                       size_t count) const {
  NGRAPH_CHECK(count != 0, "Decode called on 0 elements");
  NGRAPH_CHECK(type == element::f32,
               "CKKS encode supports only float encoding, received type ",
               type);
  NGRAPH_CHECK(input.num_values() > 0, "Input has no values");

  const std::vector<float>& xs_float = input.values();
  NGRAPH_CHECK(xs_float.size() >= count);
  std::memcpy(output, &xs_float[0], type.size() * count);
}

void ngraph::he::HESealBackend::decode(
    ngraph::he::HEPlaintext& output,
    const ngraph::he::SealPlaintextWrapper& input) const {
  std::vector<double> real_vals;
  if (input.complex_packing()) {
    std::vector<std::complex<double>> complex_vals;
    m_ckks_encoder->decode(input.plaintext(), complex_vals);
    complex_vec_to_real_vec(real_vals, complex_vals);
  } else {
    m_ckks_encoder->decode(input.plaintext(), real_vals);
  }
  std::vector<float> float_vals{real_vals.begin(), real_vals.end()};
  output.values() = float_vals;
}

void ngraph::he::HESealBackend::encode(
    ngraph::he::SealPlaintextWrapper& destination,
    const ngraph::he::HEPlaintext& plaintext, seal::parms_id_type parms_id,
    double scale, bool complex_packing) const {
  std::vector<double> double_vals(plaintext.values().begin(),
                                  plaintext.values().end());
  const size_t slot_count = m_ckks_encoder->slot_count();

  if (complex_packing) {
    std::vector<std::complex<double>> complex_vals;
    if (double_vals.size() == 1) {
      std::complex<double> val(double_vals[0], double_vals[0]);
      complex_vals = std::vector<std::complex<double>>(slot_count, val);
    } else {
      real_vec_to_complex_vec(complex_vals, double_vals);
    }
    NGRAPH_CHECK(complex_vals.size() <= slot_count, "Cannot encode ",
                 complex_vals.size(), " elements, maximum size is ",
                 slot_count);
    m_ckks_encoder->encode(complex_vals, parms_id, scale,
                           destination.plaintext());
  } else {
    // TODO: why different cases?
    if (double_vals.size() == 1) {
      m_ckks_encoder->encode(double_vals[0], parms_id, scale,
                             destination.plaintext());
    } else {
      NGRAPH_CHECK(double_vals.size() <= slot_count, "Cannot encode ",
                   double_vals.size(), " elements, maximum size is ",
                   slot_count);
      m_ckks_encoder->encode(double_vals, parms_id, scale,
                             destination.plaintext());
    }
  }
  destination.complex_packing() = complex_packing;
}

void ngraph::he::HESealBackend::encode(
    ngraph::he::SealPlaintextWrapper& destination,
    const ngraph::he::HEPlaintext& plaintext, bool complex_packing) const {
  double scale = m_scale;
  auto parms_id = m_context->first_parms_id();

  encode(destination, plaintext, parms_id, scale, complex_packing);
}
