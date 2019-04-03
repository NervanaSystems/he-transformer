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

// TODO: remove thread and chrono
#include <boost/asio.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "seal/he_seal_client.hpp"
#include "seal/he_seal_util.hpp"
#include "seal/seal.h"
#include "tcp/tcp_client.hpp"
#include "tcp/tcp_message.hpp"

using namespace ngraph;
using namespace std;

runtime::he::HESealClient::HESealClient(const std::string& hostname,
                                        const size_t port,
                                        const size_t batch_size,
                                        const std::vector<float>& inputs)
    : m_batch_size{batch_size}, m_inputs{inputs}, m_is_done(false) {
  boost::asio::io_context io_context;
  tcp::resolver resolver(io_context);
  auto endpoints = resolver.resolve(hostname, std::to_string(port));

  auto client_callback = [this](const runtime::he::TCPMessage& message) {
    return handle_message(message);
  };

  m_tcp_client = std::make_shared<runtime::he::TCPClient>(io_context, endpoints,
                                                          client_callback);

  io_context.run();
}

void runtime::he::HESealClient::set_seal_context() {
  m_context = seal::SEALContext::Create(m_encryption_params);

  print_seal_context(*m_context);

  m_keygen = std::make_shared<seal::KeyGenerator>(m_context);
  m_relin_keys = std::make_shared<seal::RelinKeys>(m_keygen->relin_keys(60));
  m_public_key = std::make_shared<seal::PublicKey>(m_keygen->public_key());
  m_secret_key = std::make_shared<seal::SecretKey>(m_keygen->secret_key());
  m_encryptor = std::make_shared<seal::Encryptor>(m_context, *m_public_key);
  m_decryptor = std::make_shared<seal::Decryptor>(m_context, *m_secret_key);

  // Evaluator
  m_evaluator = std::make_shared<seal::Evaluator>(m_context);

  // Encoder
  m_ckks_encoder = std::make_shared<seal::CKKSEncoder>(m_context);

  auto m_context_data = m_context->context_data();
  m_scale = static_cast<double>(
      m_context_data->parms().coeff_modulus().back().value());
}

void runtime::he::HESealClient::handle_message(
    const runtime::he::TCPMessage& message) {
  runtime::he::MessageType msg_type = message.message_type();

  std::cout << "Client received message type: "
            << message_type_to_string(msg_type).c_str() << std::endl;

  if (msg_type == runtime::he::MessageType::parameter_size) {
    // Number of (packed) ciphertexts to perform inference on
    size_t parameter_size;
    std::memcpy(&parameter_size, message.data_ptr(), message.data_size());

    std::cout << "Parameter size " << parameter_size << std::endl;
    std::cout << "Client batch size " << m_batch_size << std::endl;

    std::vector<seal::Ciphertext> ciphers;
    assert(m_inputs.size() == parameter_size * m_batch_size);

    std::cout << "parameter_size " << parameter_size << std::endl;
    std::cout << "m_batch_size " << m_batch_size << std::endl;

    std::stringstream cipher_stream;
    for (size_t data_idx = 0; data_idx < parameter_size; ++data_idx) {
      seal::Plaintext plain;
      std::vector<double> encode_vals;
      for (size_t batch_idx = 0; batch_idx < m_batch_size; ++batch_idx) {
        encode_vals.emplace_back(
            (double)(m_inputs[data_idx * m_batch_size + batch_idx]));
      }
      m_ckks_encoder->encode(encode_vals, m_scale, plain);
      seal::Ciphertext c;
      m_encryptor->encrypt(plain, c);
      c.save(cipher_stream);
    }

    const std::string& cipher_str = cipher_stream.str();
    const char* cipher_cstr = cipher_str.c_str();
    size_t cipher_size = cipher_str.size();
    std::cout << "Sending execute message with " << parameter_size
              << " ciphertexts" << std::endl;
    auto execute_message = TCPMessage(runtime::he::MessageType::execute,
                                      parameter_size, cipher_size, cipher_cstr);
    write_message(execute_message);

    /* std::cout << "Waiting for 20 seconds until message sent" << std::endl;
     std::this_thread::sleep_for(std::chrono::seconds(20)); */

  } else if (msg_type == runtime::he::MessageType::result) {
    size_t result_count = message.count();
    size_t element_size = message.element_size();

    std::vector<seal::Ciphertext> result;
    m_results.reserve(result_count * m_batch_size);
    for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
      seal::Ciphertext cipher;
      std::stringstream cipher_stream;
      cipher_stream.write(message.data_ptr() + result_idx * element_size,
                          element_size);
      cipher.load(m_context, cipher_stream);

      result.push_back(cipher);
      seal::Plaintext plain;
      m_decryptor->decrypt(cipher, plain);
      std::vector<double> output;
      m_ckks_encoder->decode(plain, output);

      assert(m_batch_size <= output.size());

      for (size_t batch_idx = 0; batch_idx < m_batch_size; ++batch_idx) {
        m_results.emplace_back((float)output[batch_idx]);
      }
    }

    close_connection();
  } else if (msg_type == runtime::he::MessageType::none) {
    close_connection();
  } else if (msg_type == runtime::he::MessageType::encryption_parameters) {
    std::stringstream param_stream;
    param_stream.write(message.data_ptr(), message.element_size());
    m_encryption_params = seal::EncryptionParameters::Load(param_stream);
    std::cout << "Loaded encryption parmeters" << std::endl;

    set_seal_context();

    std::stringstream evk_stream;
    m_relin_keys->save(evk_stream);
    const std::string& evk_str = evk_stream.str();
    const char* evk_cstr = evk_str.c_str();
    size_t m_data_size = evk_str.size();
    auto evk_message = TCPMessage(runtime::he::MessageType::eval_key, 1,
                                  m_data_size, evk_cstr);
    std::cout << "Sending evaluation key" << std::endl;

    write_message(evk_message);

  } else {
    std::cout << "Unsupported message type"
              << message_type_to_string(msg_type).c_str() << std::endl;
  }
}

void runtime::he::HESealClient::write_message(
    const runtime::he::TCPMessage& message) {
  m_tcp_client->write_message(message);
}

bool runtime::he::HESealClient::is_done() { return m_is_done; }

std::vector<float> runtime::he::HESealClient::get_results() {
  return m_results;
}

void runtime::he::HESealClient::close_connection() {
  std::cout << "Closing connection" << std::endl;
  m_tcp_client->close();
  m_is_done = true;
}