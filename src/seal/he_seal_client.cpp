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

#include <algorithm>
#include <boost/asio.hpp>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "he_seal_cipher_tensor.hpp"
#include "ngraph/log.hpp"
#include "nlohmann/json.hpp"
#include "seal/he_seal_client.hpp"
#include "seal/kernel/bounded_relu_seal.hpp"
#include "seal/kernel/max_pool_seal.hpp"
#include "seal/kernel/relu_seal.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"
#include "tcp/tcp_client.hpp"
#include "tcp/tcp_message.hpp"

using json = nlohmann::json;

ngraph::he::HESealClient::HESealClient(const std::string& hostname,
                                       const size_t port,
                                       const size_t batch_size,
                                       const std::vector<double>& inputs,
                                       bool complex_packing)
    : m_batch_size{batch_size},
      m_is_done(false),
      m_inputs{inputs},
      m_complex_packing(complex_packing) {
  boost::asio::io_context io_context;
  tcp::resolver resolver(io_context);
  auto endpoints = resolver.resolve(hostname, std::to_string(port));
  auto client_callback = [this](const ngraph::he::TCPMessage& message) {
    return handle_message(message);
  };
  m_tcp_client = std::make_shared<ngraph::he::TCPClient>(io_context, endpoints,
                                                         client_callback);
  io_context.run();
}

ngraph::he::HESealClient::HESealClient(const std::string& hostname,
                                       const size_t port,
                                       const size_t batch_size,
                                       const std::vector<float>& inputs,
                                       bool complex_packing)
    : HESealClient(hostname, port, batch_size,
                   std::vector<double>(inputs.begin(), inputs.end()),
                   complex_packing) {}

void ngraph::he::HESealClient::set_seal_context() {
  m_context = seal::SEALContext::Create(m_encryption_params, true,
                                        seal::sec_level_type::none);

  print_seal_context(*m_context);

  m_keygen = std::make_shared<seal::KeyGenerator>(m_context);
  m_relin_keys = std::make_shared<seal::RelinKeys>(m_keygen->relin_keys());
  m_public_key = std::make_shared<seal::PublicKey>(m_keygen->public_key());
  m_secret_key = std::make_shared<seal::SecretKey>(m_keygen->secret_key());
  m_encryptor = std::make_shared<seal::Encryptor>(m_context, *m_public_key);
  m_decryptor = std::make_shared<seal::Decryptor>(m_context, *m_secret_key);

  // Evaluator
  m_evaluator = std::make_shared<seal::Evaluator>(m_context);

  // Encoder
  m_ckks_encoder = std::make_shared<seal::CKKSEncoder>(m_context);

  // TODO: pick better scale?
  m_scale = ngraph::he::choose_scale(m_encryption_params.coeff_modulus());
  NGRAPH_INFO << "Client scale " << m_scale;
}

void ngraph::he::HESealClient::send_public_and_relin_keys() {
  NGRAPH_INFO << "SEnding public and relin keys";

  he_proto::TCPMessage proto_msg;
  proto_msg.set_type(he_proto::TCPMessage_Type_RESPONSE);

  // Set public key
  std::stringstream pk_stream;
  m_public_key->save(pk_stream);
  he_proto::PublicKey public_key;
  public_key.set_public_key(pk_stream.str());
  *proto_msg.mutable_public_key() = public_key;

  // Set relinearization keys
  std::stringstream evk_stream;
  m_relin_keys->save(evk_stream);
  he_proto::EvaluationKey eval_key;
  eval_key.set_eval_key(evk_stream.str());
  *proto_msg.mutable_eval_key() = eval_key;

  NGRAPH_INFO << "Sending pk / evk";
  write_message(proto_msg);
}

void ngraph::he::HESealClient::handle_encryption_parameters_response(
    const he_proto::TCPMessage& proto_msg) {
  NGRAPH_INFO << "Got enc parms request";
  NGRAPH_CHECK(proto_msg.has_encryption_parameters(),
               "proto_msg does not have encryption_parameters");

  const std::string& enc_parms_str =
      proto_msg.encryption_parameters().encryption_parameters();
  std::stringstream param_stream(enc_parms_str);
  m_encryption_params = seal::EncryptionParameters::Load(param_stream);

  NGRAPH_INFO << "Loaded enc parms";

  set_seal_context();
  send_public_and_relin_keys();
}

void ngraph::he::HESealClient::handle_inference_request(
    const he_proto::TCPMessage& proto_msg) {
  NGRAPH_INFO << "handle_inference_request";
  NGRAPH_CHECK(proto_msg.has_function(), "Proto msg doesn't have funtion");

  const std::string& inference_shape = proto_msg.function().function();
  json js = json::parse(inference_shape);
  std::vector<size_t> shape_dims = js.at("shape");
  ngraph::Shape shape{shape_dims};

  NGRAPH_INFO << join(shape, "x");

  size_t parameter_size = ngraph::shape_size(shape);

  NGRAPH_INFO << "Parameter size " << parameter_size;
  NGRAPH_INFO << "Client batch size " << m_batch_size;
  NGRAPH_INFO << "m_inputs.size() " << m_inputs.size();
  if (complex_packing()) {
    NGRAPH_INFO << "Client complex packing";
  }

  if (m_inputs.size() > parameter_size * m_batch_size) {
    NGRAPH_INFO << "m_inputs.size() " << m_inputs.size()
                << " > paramter_size ( " << parameter_size
                << ") * m_batch_size (" << m_batch_size << ")";
  }

  std::vector<std::shared_ptr<SealCiphertextWrapper>> ciphers(parameter_size /
                                                              m_batch_size);
  for (size_t data_idx = 0; data_idx < ciphers.size(); ++data_idx) {
    ciphers[data_idx] = std::make_shared<SealCiphertextWrapper>();
  }

  for (const auto& elem : m_inputs) {
    NGRAPH_INFO << "input " << elem;
  }

  // TODO: add element type to function message
  size_t num_bytes = parameter_size * sizeof(double);
  ngraph::he::HESealCipherTensor::write(
      ciphers, m_inputs.data(), num_bytes, m_batch_size, element::f64,
      m_context->first_parms_id(), m_scale, *m_ckks_encoder, *m_encryptor,
      complex_packing());

  he_proto::TCPMessage encrypted_inputs_msg;
  encrypted_inputs_msg.set_type(he_proto::TCPMessage_Type_REQUEST);

  for (size_t data_idx = 0; data_idx < ciphers.size(); ++data_idx) {
    he_proto::SealCiphertextWrapper* proto_cipher =
        encrypted_inputs_msg.add_ciphers();
    proto_cipher->set_known_value(false);
    // TODO: save directly to protobuf
    std::stringstream s;
    ciphers[data_idx]->ciphertext().save(s);
    proto_cipher->set_ciphertext(s.str());
  }

  NGRAPH_INFO << "Creating execute message";
  write_message(encrypted_inputs_msg);
}

void ngraph::he::HESealClient::handle_result(
    const he_proto::TCPMessage& proto_msg) {
  NGRAPH_INFO << "handling result";

  size_t result_count = proto_msg.ciphers_size();
  m_results.resize(result_count * m_batch_size);
  std::vector<std::shared_ptr<SealCiphertextWrapper>> result_ciphers(
      result_count);
#pragma omp parallel for
  for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
    auto proto_cipher = proto_msg.ciphers(result_idx);

    if (proto_cipher.known_value()) {
      NGRAPH_INFO << "Loading known-valued cipher in result with value "
                  << proto_cipher.value();

      auto cipher_wrapper = std::make_shared<SealCiphertextWrapper>();
      cipher_wrapper->known_value() = true;
      cipher_wrapper->value() = proto_cipher.value();

      result_ciphers[result_idx] = cipher_wrapper;

    } else {
      seal::Ciphertext c;
      // TODO: load from string directly
      const std::string& cipher_str =
          proto_msg.ciphers(result_idx).ciphertext();
      std::stringstream ss;
      ss.str(cipher_str);
      c.load(m_context, ss);

      result_ciphers[result_idx] =
          std::make_shared<SealCiphertextWrapper>(c, complex_packing());
    }
  }

  size_t n = result_count * sizeof(double) * m_batch_size;
  ngraph::he::HESealCipherTensor::read(
      m_results.data(), result_ciphers, n, m_batch_size, element::f64,
      m_context->first_parms_id(), m_scale, *m_ckks_encoder, *m_decryptor,
      complex_packing());

  NGRAPH_INFO << "done handling result";

  close_connection();
}

void ngraph::he::HESealClient::handle_relu_request(
    const he_proto::TCPMessage& proto_msg) {
  NGRAPH_INFO << "Handling relu request";
  NGRAPH_CHECK(proto_msg.has_function(), "Proto message doesn't have function");

  he_proto::TCPMessage proto_relu;
  proto_relu.set_type(he_proto::TCPMessage_Type_RESPONSE);
  *proto_relu.mutable_function() = proto_msg.function();

  size_t result_count = proto_msg.ciphers_size();

  NGRAPH_INFO << "result_count " << result_count;
  //#pragma omp parallel for
  for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
    NGRAPH_CHECK(!proto_msg.ciphers(result_idx).known_value(),
                 "Client should not receive known-valued relu values");

    auto post_relu_cipher = std::make_shared<SealCiphertextWrapper>();

    seal::Ciphertext pre_relu_cipher;
    // TODO: load from string directly
    const std::string& cipher_str = proto_msg.ciphers(result_idx).ciphertext();
    std::stringstream ss;
    ss.str(cipher_str);
    pre_relu_cipher.load(m_context, ss);
    SealCiphertextWrapper wrapped_cipher(pre_relu_cipher, complex_packing());

    ngraph::he::scalar_relu_seal(wrapped_cipher, post_relu_cipher,
                                 m_context->first_parms_id(), m_scale,
                                 *m_ckks_encoder, *m_encryptor, *m_decryptor);

    he_proto::SealCiphertextWrapper* proto_cipher = proto_relu.add_ciphers();
    proto_cipher->set_known_value(false);
    // TODO: save directly to protobuf
    std::stringstream s;
    post_relu_cipher->ciphertext().save(s);
    proto_cipher->set_ciphertext(s.str());
  }

  ngraph::he::TCPMessage relu_result_msg(proto_relu);

  NGRAPH_INFO << "Writing relu result";
  write_message(relu_result_msg);
  return;
}

void ngraph::he::HESealClient::handle_bounded_relu_request(
    const he_proto::TCPMessage& proto_msg) {
  NGRAPH_INFO << "Handling bounded_relu request";

  /* ngraph::he::scalar_bounded_relu_seal(
      wrapped_cipher, post_relu_ciphers[result_idx], 6.0f,
      m_context->first_parms_id(), m_scale, *m_ckks_encoder, *m_encryptor,
      *m_decryptor); */
}

void ngraph::he::HESealClient::handle_max_pool_request(
    const he_proto::TCPMessage& proto_msg) {
  NGRAPH_INFO << "Handling maxpool request";
  NGRAPH_CHECK(proto_msg.has_function(), "Proto message doesn't have function");

  size_t cipher_count = proto_msg.ciphers_size();
  std::vector<std::shared_ptr<SealCiphertextWrapper>> maxpool_ciphers(
      cipher_count);
  std::vector<std::shared_ptr<SealCiphertextWrapper>> post_max_cipher(1);
  post_max_cipher[0] = std::make_shared<SealCiphertextWrapper>();

#pragma omp parallel for
  for (size_t cipher_idx = 0; cipher_idx < cipher_count; ++cipher_idx) {
    seal::Ciphertext pre_max_pool_cipher;
    maxpool_ciphers[cipher_idx] = std::make_shared<SealCiphertextWrapper>();

    // TODO: load from string directly
    const std::string& cipher_str = proto_msg.ciphers(cipher_idx).ciphertext();
    std::stringstream ss;
    ss.str(cipher_str);
    pre_max_pool_cipher.load(m_context, ss);

    maxpool_ciphers[cipher_idx]->ciphertext() = pre_max_pool_cipher;
    maxpool_ciphers[cipher_idx]->complex_packing() = complex_packing();
  }

  // We currently just support maxpool with single output
  ngraph::he::max_pool_seal(
      maxpool_ciphers, post_max_cipher, Shape{1, 1, cipher_count},
      Shape{1, 1, 1}, Shape{cipher_count}, ngraph::Strides{1}, Shape{0},
      Shape{0}, m_context->first_parms_id(), m_scale, *m_ckks_encoder,
      *m_encryptor, *m_decryptor, complex_packing());

  // Create maxpool result message
  he_proto::TCPMessage proto_max_pool;
  proto_max_pool.set_type(he_proto::TCPMessage_Type_RESPONSE);
  *proto_max_pool.mutable_function() = proto_msg.function();

  // TODO: replace with function
  auto result_cipher_wrapper = post_max_cipher[0];
  he_proto::SealCiphertextWrapper* proto_cipher = proto_max_pool.add_ciphers();
  proto_cipher->set_known_value(result_cipher_wrapper->known_value());
  if (result_cipher_wrapper->known_value()) {
    proto_cipher->set_value(result_cipher_wrapper->value());
  } else {
    // TODO: write directly to ciphertext
    std::stringstream ss;
    result_cipher_wrapper->ciphertext().save(ss);
    proto_cipher->set_ciphertext(ss.str());
  }

  ngraph::he::TCPMessage max_pool_result_msg(proto_max_pool);

  NGRAPH_INFO << "Writing maxpool result";
  write_message(max_pool_result_msg);
  return;
}

void ngraph::he::HESealClient::handle_message(
    const ngraph::he::TCPMessage& message) {
  // TODO: try overwriting message?

  std::shared_ptr<he_proto::TCPMessage> proto_msg = message.proto_message();

  switch (proto_msg->type()) {
    case he_proto::TCPMessage_Type_RESPONSE: {
      NGRAPH_INFO << "Client got message RESPONSE";
      if (proto_msg->has_encryption_parameters()) {
        handle_encryption_parameters_response(*proto_msg);
      } else if (proto_msg->ciphers_size() > 0) {
        handle_result(*proto_msg);
      } else {
        NGRAPH_CHECK(false, "Unknown RESPONSE type");
      }
      break;
    }
    case he_proto::TCPMessage_Type_REQUEST: {
      NGRAPH_INFO << "Client got message REQUEST";

      if (proto_msg->has_function()) {
        const std::string& function = proto_msg->function().function();
        json js = json::parse(function);

        auto name = js.at("function");
        if (name == "Parameter") {
          handle_inference_request(*proto_msg);
        } else if (name == "Relu") {
          handle_relu_request(*proto_msg);
        } else if (name == "Bounded_Relu") {
          handle_bounded_relu_request(*proto_msg);
        } else if (name == "MaxPool") {
          handle_max_pool_request(*proto_msg);
        } else {
          NGRAPH_INFO << "Unknown name " << name;
        }
      } else {
        NGRAPH_CHECK(false, "Unknown REQUEST type");
      }

      break;
    }
    case he_proto::TCPMessage_Type_UNKNOWN:
    default:
      NGRAPH_CHECK(false, "Unknonwn TCPMesage type");
  }
}

void ngraph::he::HESealClient::close_connection() {
  NGRAPH_INFO << "Closing connection";
  m_tcp_client->close();
  m_is_done = true;
}
