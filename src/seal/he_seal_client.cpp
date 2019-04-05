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
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
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
    : m_is_done(false), m_batch_size{batch_size}, m_inputs{inputs} {
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

    // Send public key
    std::stringstream pk_stream;
    m_public_key->save(pk_stream);
    auto pk_message =
        TCPMessage(runtime::he::MessageType::public_key, 1, pk_stream);
    std::cout << "Sending public key" << std::endl;
    write_message(pk_message);

    // Send evaluation key
    std::stringstream evk_stream;
    m_relin_keys->save(evk_stream);
    auto evk_message =
        TCPMessage(runtime::he::MessageType::eval_key, 1, evk_stream);
    std::cout << "Sending evaluation key" << std::endl;
    write_message(evk_message);

  } else if (msg_type == runtime::he::MessageType::relu_request) {
    size_t result_count = message.count();
    size_t element_size = message.element_size();

    std::stringstream post_relu_stream;
    std::vector<seal::Ciphertext> post_relu_ciphers(result_count);
#pragma omp parallel for
    for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
      seal::Ciphertext pre_relu_cipher;
      seal::Plaintext pre_relu_plain;
      seal::Plaintext post_relu_plain;

      // Load cipher from stream
      std::stringstream pre_relu_cipher_stream;
      pre_relu_cipher_stream.write(
          message.data_ptr() + result_idx * element_size, element_size);
      pre_relu_cipher.load(m_context, pre_relu_cipher_stream);

      // Decrypt cipher
      m_decryptor->decrypt(pre_relu_cipher, pre_relu_plain);
      std::vector<double> pre_relu;
      m_ckks_encoder->decode(pre_relu_plain, pre_relu);

      // Perform relu
      std::vector<double> post_relu(m_batch_size);
      for (size_t batch_idx = 0; batch_idx < m_batch_size; ++batch_idx) {
        double pre_relu_val = pre_relu[batch_idx];
        double post_relu_val = pre_relu_val > 0 ? pre_relu_val : 0;
        // std::cout << "relu(" << pre_relu_val << ") = " << post_relu_val
        //          << std::endl;
        post_relu[batch_idx] = post_relu_val;
      }

      // Encrypt post-relu result
      m_ckks_encoder->encode(post_relu, m_scale, post_relu_plain);
      m_encryptor->encrypt(post_relu_plain, post_relu_ciphers[result_idx]);
    }
    for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
      post_relu_ciphers[result_idx].save(post_relu_stream);
    }
    std::cout << "Writing relu_result message with " << result_count
              << " ciphertexts" << std::endl;

    auto relu_result_msg = TCPMessage(runtime::he::MessageType::relu_result,
                                      result_count, post_relu_stream);
    write_message(relu_result_msg);
  } else if (msg_type == runtime::he::MessageType::max_request) {
    size_t cipher_count = message.count();
    size_t element_size = message.element_size();

    std::vector<std::vector<double>> input_cipher_values(
        m_batch_size, vector<double>(cipher_count, 0));

    std::vector<double> max_values(m_batch_size,
                                   std::numeric_limits<double>::lowest());

#pragma omp parallel for
    for (size_t cipher_idx = 0; cipher_idx < cipher_count; ++cipher_idx) {
      seal::Ciphertext pre_sort_cipher;
      seal::Plaintext pre_sort_plain;

      // Load cipher from stream
      std::stringstream pre_sort_cipher_stream;
      pre_sort_cipher_stream.write(
          message.data_ptr() + cipher_idx * element_size, element_size);
      pre_sort_cipher.load(m_context, pre_sort_cipher_stream);

      // Decrypt cipher
      m_decryptor->decrypt(pre_sort_cipher, pre_sort_plain);
      std::vector<double> pre_sort_value;
      m_ckks_encoder->decode(pre_sort_plain, pre_sort_value);

      // Discard extra values
      pre_sort_value.resize(m_batch_size);

      for (size_t batch_idx = 0; batch_idx < m_batch_size; ++batch_idx) {
        input_cipher_values[batch_idx][cipher_idx] = pre_sort_value[batch_idx];
      }
    }

    // Get max of each vector of values
    for (size_t batch_idx = 0; batch_idx < m_batch_size; ++batch_idx) {
      max_values[batch_idx] =
          *std::max_element(input_cipher_values[batch_idx].begin(),
                            input_cipher_values[batch_idx].end());
    }

    // Encrypt maximum values
    seal::Ciphertext cipher_max;
    seal::Plaintext plain_max;
    std::stringstream max_stream;
    m_ckks_encoder->encode(max_values, m_scale, plain_max);
    m_encryptor->encrypt(plain_max, cipher_max);
    cipher_max.save(max_stream);
    std::cout << "Writing max_result message with " << 1 << " ciphertexts"
              << std::endl;

    auto max_result_msg =
        TCPMessage(runtime::he::MessageType::max_result, 1, max_stream);
    write_message(max_result_msg);
  } else {
    std::cout << "Unsupported message type: "
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