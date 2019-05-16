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

  // std::cout << "Client received message type: "
  //          << message_type_to_string(msg_type).c_str() << std::endl;

  if (msg_type == runtime::he::MessageType::parameter_size) {
    // Number of (packed) ciphertexts to perform inference on
    size_t parameter_size;
    std::memcpy(&parameter_size, message.data_ptr(), message.data_size());

    std::cout << "Parameter size " << parameter_size << std::endl;
    std::cout << "Client batch size " << m_batch_size << std::endl;
    std::cout << "Client packing? " << complex_packing();

    // TODO: support odd batch sizes
    assert(m_batch_size % 2 == 0);

    std::vector<seal::Ciphertext> ciphers;
    if (complex_packing()) {
      std::cout << "m_inputs.size() " << m_inputs.size() << std::endl;
      std::cout << "parameter_size " << parameter_size << std::endl;
      std::cout << "m_batch_size " << m_batch_size << std::endl;
      assert(m_inputs.size() == parameter_size * m_batch_size * 2);
    } else {
      assert(m_inputs.size() == parameter_size * m_batch_size);
    }
    /*for (auto& elem : m_inputs) {
      std::cout << elem << std::endl;
    }*/

    std::stringstream cipher_stream;
    for (size_t data_idx = 0; data_idx < parameter_size; ++data_idx) {
      seal::Plaintext plain;
      if (complex_packing()) {
        size_t complex_scale_factor = 2;
        std::vector<complex<double>> encode_vals;
        size_t batch_start_idx = data_idx * m_batch_size * complex_scale_factor;
        double real_part = 0;
        double imag_part = 0;

        for (size_t batch_idx = 0;
             batch_idx < m_batch_size * complex_scale_factor; ++batch_idx) {
          if (batch_idx % 2 == 0) {
            real_part = m_inputs[batch_start_idx + batch_idx];
          } else {
            imag_part = m_inputs[batch_start_idx + batch_idx];
            std::complex<double> encode_val(real_part, imag_part);
            std::cout << "encode val " << encode_val << std::endl;
            encode_vals.emplace_back(encode_val);
            imag_part = 0;
            real_part = 0;
          }
        }
        std::cout << "Encoding encode vals" << std::endl;
        m_ckks_encoder->encode(encode_vals, m_scale, plain);
      } else {
        std::vector<double> encode_vals;
        for (size_t batch_idx = 0; batch_idx < m_batch_size; ++batch_idx) {
          encode_vals.emplace_back(
              (double)(m_inputs[data_idx * m_batch_size + batch_idx]));
        }
        m_ckks_encoder->encode(encode_vals, m_scale, plain);
      }
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

    std::cout << "Client got " << result_count << " results " << std::endl;

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

      if (complex_packing()) {
        std::cout << std::endl
                  << "Got complex result " << result_idx << std::endl;
        std::vector<complex<double>> outputs;
        m_ckks_encoder->decode(plain, outputs);

        std::cout << "outputs.size() " << outputs.size() << std::endl;
        std::cout << "m_batch_size " << m_batch_size << std::endl;

        assert(m_batch_size <= outputs.size());

        for (size_t batch_idx = 0;
             batch_idx < m_batch_size /* outputs.size() */; ++batch_idx) {
          float re = (float)outputs[batch_idx].real();
          float im = (float)outputs[batch_idx].imag();
          m_results.emplace_back(re);
          m_results.emplace_back(im);

          std::cout << "re " << re << " imag " << im << std::endl;
        }
      } else {
        std::vector<double> outputs;
        m_ckks_encoder->decode(plain, outputs);

        assert(m_batch_size <= outputs.size());

        for (size_t batch_idx = 0; batch_idx < m_batch_size; ++batch_idx) {
          m_results.emplace_back((float)outputs[batch_idx]);
        }
      }
    }
    std::cout << "Results size " << m_results.size() << std::endl;

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

      if (complex_packing()) {
        std::vector<complex<double>> pre_relu;
        m_ckks_encoder->decode(pre_relu_plain, pre_relu);
        // Perform relu
        std::vector<complex<double>> post_relu(m_batch_size);
        for (size_t batch_idx = 0; batch_idx < m_batch_size; ++batch_idx) {
          complex<double> pre_relu_val = pre_relu[batch_idx];

          double post_relu_real =
              pre_relu_val.real() > 0 ? pre_relu_val.real() : 0;
          double post_relu_imag =
              pre_relu_val.imag() > 0 ? pre_relu_val.imag() : 0;

          complex<double> post_relu_value(post_relu_real, post_relu_imag);

          // std::cout << "relu(" << pre_relu_val << ") = " << post_relu_value
          //          << std::endl;
          post_relu[batch_idx] = post_relu_value;
        }
        m_ckks_encoder->encode(post_relu, m_scale, post_relu_plain);
      } else {
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

        m_ckks_encoder->encode(post_relu, m_scale, post_relu_plain);
      }
      // Encrypt post-relu result
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
    size_t complex_scale_factor = complex_packing() ? 2 : 1;
    size_t cipher_count = message.count();

    size_t element_size = message.element_size();

    std::vector<std::vector<double>> input_cipher_values(
        m_batch_size * complex_scale_factor, vector<double>(cipher_count, 0));

    std::vector<double> max_values(m_batch_size * complex_scale_factor,
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
      if (complex_packing()) {
        std::vector<complex<double>> complex_pre_sort_vals;
        m_ckks_encoder->decode(pre_sort_plain, complex_pre_sort_vals);

        for (const auto& val : complex_pre_sort_vals) {
          pre_sort_value.emplace_back(val.real());
          pre_sort_value.emplace_back(val.imag());
        }
      } else {
        m_ckks_encoder->decode(pre_sort_plain, pre_sort_value);
      }

      // Discard extra values
      pre_sort_value.resize(m_batch_size * complex_scale_factor);

      for (size_t batch_idx = 0;
           batch_idx < m_batch_size * complex_scale_factor; ++batch_idx) {
        input_cipher_values[batch_idx][cipher_idx] = pre_sort_value[batch_idx];
      }
    }

    // Get max of each vector of values
    for (size_t batch_idx = 0; batch_idx < m_batch_size * complex_scale_factor;
         ++batch_idx) {
      max_values[batch_idx] =
          *std::max_element(input_cipher_values[batch_idx].begin(),
                            input_cipher_values[batch_idx].end());
    }

    // Encrypt maximum values
    seal::Ciphertext cipher_max;
    seal::Plaintext plain_max;
    std::stringstream max_stream;

    if (complex_packing()) {
      std::cout << "max_values size " << max_values.size() << std::endl;
      assert(max_values.size() % 2 == 0);
      std::vector<std::complex<double>> max_complex_vals;
      for (size_t max_idx = 0; max_idx < max_values.size() / 2; max_idx += 2) {
        assert(2 * max_idx + 1 < max_values.size());
        double re = max_values[2 * max_idx];
        double imag = max_values[2 * max_idx + 1];
        max_complex_vals.push_back(std::complex<double>(re, imag));
      }
      m_ckks_encoder->encode(max_complex_vals, m_scale, plain_max);

    } else {
      m_ckks_encoder->encode(max_values, m_scale, plain_max);
    }
    m_encryptor->encrypt(plain_max, cipher_max);
    cipher_max.save(max_stream);
    // std::cout << "Writing max_result message with " << 1 << " ciphertexts"
    //          << std::endl;

    auto max_result_msg =
        TCPMessage(runtime::he::MessageType::max_result, 1, max_stream);
    write_message(max_result_msg);
  } else if (msg_type == runtime::he::MessageType::minimum_request) {
    // Stores (c_1a, c_1b, c_2a, c_b, ..., c_na, c_nb)
    // prints mesage (min(c_1a, c_1b), min(c_2a, c_2b), ..., min(c_na, c_nb))
    size_t cipher_count = message.count();
    assert(cipher_count % 2 == 0);
    size_t element_size = message.element_size();

    std::vector<std::vector<double>> input_cipher_values(
        cipher_count, vector<double>(m_batch_size, 0));

    std::vector<double> min_values(m_batch_size,
                                   std::numeric_limits<double>::max());

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
        input_cipher_values[cipher_idx][batch_idx] = pre_sort_value[batch_idx];
      }
    }

    // Get minimum of each vector of values
    std::stringstream minimum_stream;
    for (size_t cipher_idx = 0; cipher_idx < cipher_count; cipher_idx += 2) {
      for (size_t batch_idx = 0; batch_idx < m_batch_size; ++batch_idx) {
        min_values[batch_idx] =
            min(input_cipher_values[cipher_idx][batch_idx],
                input_cipher_values[cipher_idx + 1][batch_idx]);
      }
      // Encrypt minimum values
      seal::Ciphertext cipher_minimum;
      seal::Plaintext plain_minimum;
      m_ckks_encoder->encode(min_values, m_scale, plain_minimum);
      m_encryptor->encrypt(plain_minimum, cipher_minimum);
      cipher_minimum.save(minimum_stream);
    }

    // std::cout << "Writing minimum_result message with " << (cipher_count /
    // 2)
    //          << " ciphertexts" << std::endl;

    auto minimum_result_msg =
        TCPMessage(runtime::he::MessageType::minimum_result, cipher_count / 2,
                   minimum_stream);
    write_message(minimum_result_msg);
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