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
#include "seal/he_seal_client.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"
#include "tcp/tcp_client.hpp"
#include "tcp/tcp_message.hpp"

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

void ngraph::he::HESealClient::handle_message(
    const ngraph::he::TCPMessage& message) {
  ngraph::he::MessageType msg_type = message.message_type();

  NGRAPH_DEBUG << "Client received message type: " << msg_type;

  switch (msg_type) {
    case ngraph::he::MessageType::parameter_size: {
      // Number of (packed) ciphertexts to perform inference on
      size_t parameter_size;
      std::memcpy(&parameter_size, message.data_ptr(), message.data_size());

      NGRAPH_INFO << "Parameter size " << parameter_size;
      NGRAPH_INFO << "Client batch size " << m_batch_size;
      if (complex_packing()) {
        NGRAPH_INFO << "Client complex packing";
      }

      if (m_inputs.size() > parameter_size * m_batch_size) {
        NGRAPH_INFO << "m_inputs.size() " << m_inputs.size()
                    << " > paramter_size ( " << parameter_size
                    << ") * m_batch_size (" << m_batch_size << ")";
      }

      std::vector<std::shared_ptr<SealCiphertextWrapper>> ciphers(
          parameter_size);
      for (size_t data_idx = 0; data_idx < parameter_size; ++data_idx) {
        ciphers[data_idx] = std::make_shared<SealCiphertextWrapper>();
      }

      size_t n = parameter_size * sizeof(double) * m_batch_size;
      ngraph::he::HESealCipherTensor::write(
          ciphers, m_inputs.data(), n, m_batch_size, element::f64,
          m_context->first_parms_id(), m_scale, *m_ckks_encoder, *m_encryptor,
          complex_packing());

      NGRAPH_INFO << "Creating execute message";
      auto execute_message =
          TCPMessage(ngraph::he::MessageType::execute, ciphers);
      NGRAPH_INFO << "Sending execute message with " << parameter_size
                  << " ciphertexts";
      write_message(std::move(execute_message));
      break;
    }
    case ngraph::he::MessageType::result: {
      NGRAPH_INFO << "Got result message";
      print_seal_context(*m_context);

      size_t result_count = message.count();
      m_results.resize(result_count * m_batch_size);
      std::vector<std::shared_ptr<SealCiphertextWrapper>> result_ciphers(
          result_count);
#pragma omp parallel for
      for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
        seal::Ciphertext c;
        message.load_cipher(c, result_idx, m_context);
        result_ciphers[result_idx] =
            std::make_shared<SealCiphertextWrapper>(c, complex_packing());
      }
      NGRAPH_INFO << "Loaded results; reading to m_results";

      size_t n = result_count * sizeof(double) * m_batch_size;
      ngraph::he::HESealCipherTensor::read(
          m_results.data(), result_ciphers, n, m_batch_size, element::f64,
          m_context->first_parms_id(), m_scale, *m_ckks_encoder, *m_decryptor,
          complex_packing());

      close_connection();
      break;
    }

    case ngraph::he::MessageType::none: {
      close_connection();
      break;
    }

    case ngraph::he::MessageType::encryption_parameters: {
      std::stringstream param_stream;
      param_stream.write(message.data_ptr(), message.element_size());
      m_encryption_params = seal::EncryptionParameters::Load(param_stream);
      NGRAPH_INFO << "Loaded encryption parmeters";

      set_seal_context();

      std::stringstream pk_stream;
      m_public_key->save(pk_stream);
      auto pk_message = TCPMessage(ngraph::he::MessageType::public_key, 1,
                                   std::move(pk_stream));
      NGRAPH_INFO << "Sending public key";
      write_message(std::move(pk_message));

      std::stringstream evk_stream;
      m_relin_keys->save(evk_stream);
      auto evk_message = TCPMessage(ngraph::he::MessageType::eval_key, 1,
                                    std::move(evk_stream));
      NGRAPH_INFO << "Sending evaluation key";
      write_message(std::move(evk_message));

      break;
    }
    case ngraph::he::MessageType::relu6_request: {
      handle_relu_request(message);
      break;
    }
    case ngraph::he::MessageType::relu_request: {
      handle_relu_request(message);
      break;
    }

    case ngraph::he::MessageType::maxpool_request: {
      size_t complex_pack_factor = complex_packing() ? 2 : 1;
      size_t cipher_count = message.count();

      std::vector<std::vector<double>> input_cipher_values(
          m_batch_size * complex_pack_factor,
          std::vector<double>(cipher_count, 0));

      std::vector<double> max_values(m_batch_size * complex_pack_factor,
                                     std::numeric_limits<double>::lowest());

#pragma omp parallel for
      for (size_t cipher_idx = 0; cipher_idx < cipher_count; ++cipher_idx) {
        seal::Ciphertext pre_sort_cipher;
        ngraph::he::HEPlaintext pre_sort_plain;

        message.load_cipher(pre_sort_cipher, cipher_idx, m_context);
        ngraph::he::decrypt(pre_sort_plain, pre_sort_cipher, complex_packing(),
                            *m_decryptor, *m_ckks_encoder);

        for (size_t batch_idx = 0;
             batch_idx < m_batch_size * complex_pack_factor; ++batch_idx) {
          input_cipher_values[batch_idx][cipher_idx] =
              pre_sort_plain.values()[batch_idx];
        }
      }

      // Get max of each vector of values
      for (size_t batch_idx = 0; batch_idx < m_batch_size * complex_pack_factor;
           ++batch_idx) {
        max_values[batch_idx] = static_cast<double>(
            *std::max_element(input_cipher_values[batch_idx].begin(),
                              input_cipher_values[batch_idx].end()));
      }
      seal::Ciphertext max_cipher;
      ngraph::he::encrypt(max_cipher, HEPlaintext(max_values),
                          m_context->first_parms_id(), ngraph::element::f32,
                          m_scale, *m_ckks_encoder, *m_encryptor,
                          complex_packing());

      auto maxpool_result_msg =
          TCPMessage(ngraph::he::MessageType::maxpool_result, max_cipher);
      write_message(std::move(maxpool_result_msg));

      break;
    }
    case ngraph::he::MessageType::execute:
    case ngraph::he::MessageType::eval_key:
    case ngraph::he::MessageType::maxpool_result:
    case ngraph::he::MessageType::minimum_request:
    case ngraph::he::MessageType::minimum_result:
    case ngraph::he::MessageType::parameter_shape_request:
    case ngraph::he::MessageType::public_key:
    case ngraph::he::MessageType::relu_result:
    case ngraph::he::MessageType::result_request:
    default:
      NGRAPH_INFO << "Unsupported message type: " << msg_type;
  }
}

void ngraph::he::HESealClient::close_connection() {
  NGRAPH_INFO << "Closing connection";
  m_tcp_client->close();
  m_is_done = true;
}

void ngraph::he::HESealClient::handle_relu_request(
    const ngraph::he::TCPMessage& message) {
  auto relu = [=](double d) { return d > 0 ? d : 0; };
  auto relu6 = [=](double d) { return d > 6.0 ? 6.0 : (d > 0) ? d : 0.; };

  std::function<double(double)> activation;

  if (message.message_type() == ngraph::he::MessageType::relu6_request) {
    activation = relu6;
  } else if (message.message_type() == ngraph::he::MessageType::relu_request) {
    activation = relu;
  } else {
    throw ngraph_error("Non-relu message type in handle_relu_request");
  }

  size_t result_count = message.count();
  size_t element_size = message.element_size();
  NGRAPH_INFO << "Received Relu request with " << result_count << " elements"
              << " of size " << element_size;

  std::vector<seal::Ciphertext> post_relu_ciphers(result_count);
#pragma omp parallel for
  for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
    seal::Ciphertext pre_relu_cipher;
    ngraph::he::HEPlaintext relu_plain;

    message.load_cipher(pre_relu_cipher, result_idx, m_context);
    ngraph::he::decrypt(relu_plain, pre_relu_cipher, complex_packing(),
                        *m_decryptor, *m_ckks_encoder);

    const std::vector<double>& relu_values = relu_plain.values();
    NGRAPH_CHECK(relu_values.size() >= m_batch_size,
                 "Not enough relu values in plaintext");
    std::vector<double> post_relu_values(relu_values.begin(),
                                         relu_values.begin() + m_batch_size);
    NGRAPH_INFO << "Relu[ " << result_idx << " ] values ";
    for (const auto& elem : post_relu_values) {
      NGRAPH_INFO << elem;
    }

    std::transform(post_relu_values.begin(), post_relu_values.end(),
                   post_relu_values.begin(), activation);
    relu_plain.set_values(post_relu_values);

    ngraph::he::encrypt(post_relu_ciphers[result_idx], relu_plain,
                        m_context->first_parms_id(), ngraph::element::f32,
                        m_scale, *m_ckks_encoder, *m_encryptor,
                        complex_packing());
  }
  auto relu_result_msg =
      TCPMessage(ngraph::he::MessageType::relu_result, post_relu_ciphers);

  write_message(std::move(relu_result_msg));
  return;
}
