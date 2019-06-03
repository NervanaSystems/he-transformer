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

#include "ngraph/log.hpp"
#include "seal/he_seal_client.hpp"
#include "seal/seal.h"
#include "seal/seal_util.hpp"
#include "tcp/tcp_client.hpp"
#include "tcp/tcp_message.hpp"

ngraph::he::HESealClient::HESealClient(const std::string& hostname,
                                       const size_t port,
                                       const size_t batch_size,
                                       const std::vector<float>& inputs)
    : m_is_done(false), m_batch_size{batch_size}, m_inputs{inputs} {
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

void ngraph::he::HESealClient::set_seal_context() {
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

void ngraph::he::HESealClient::handle_message(
    const ngraph::he::TCPMessage& message) {
  ngraph::he::MessageType msg_type = message.message_type();

  // NGRAPH_INFO << "Client received message type: "
  //          << message_type_to_string(msg_type).c_str() ;

  auto decode_to_real_vec = [this](const seal::Plaintext& plain,
                                   std::vector<double>& output, bool complex) {
    assert(output.size() == 0);
    if (complex) {
      std::vector<std::complex<double>> complex_outputs;
      m_ckks_encoder->decode(plain, complex_outputs);
      assert(complex_outputs.size() >= m_batch_size);
      complex_outputs.resize(m_batch_size);
      complex_vec_to_real_vec(output, complex_outputs);
    } else {
      m_ckks_encoder->decode(plain, output);
      assert(m_batch_size <= output.size());
      output.resize(m_batch_size);
    }
  };

  if (msg_type == ngraph::he::MessageType::parameter_size) {
    // Number of (packed) ciphertexts to perform inference on
    size_t parameter_size;
    std::memcpy(&parameter_size, message.data_ptr(), message.data_size());

    NGRAPH_INFO << "Parameter size " << parameter_size;
    NGRAPH_INFO << "Client batch size " << m_batch_size;
    if (complex_packing()) {
      NGRAPH_INFO << "Client complex packing? " << complex_packing();
    }

    if (complex_packing()) {
      // TODO: support odd batch sizes
      assert(m_batch_size % 2 == 0);

      if (m_inputs.size() != parameter_size * m_batch_size * 2) {
        NGRAPH_INFO << "m_inputs.size() " << m_inputs.size();
        NGRAPH_INFO << "parameter_size " << parameter_size;
        NGRAPH_INFO << "m_batch_size " << m_batch_size;
      }
      assert(m_inputs.size() == parameter_size * m_batch_size * 2);
    } else {
      assert(m_inputs.size() == parameter_size * m_batch_size);
    }

    std::vector<seal::Ciphertext> ciphers(parameter_size);
#pragma omp parallel for
    for (size_t data_idx = 0; data_idx < parameter_size; ++data_idx) {
      seal::Plaintext plain;

      size_t complex_scale_factor = complex_packing() ? 2 : 1;
      size_t batch_start_idx = data_idx * m_batch_size * complex_scale_factor;
      size_t batch_end_idx =
          batch_start_idx + m_batch_size * complex_scale_factor;

      std::vector<double> real_vals{m_inputs.begin() + batch_start_idx,
                                    m_inputs.begin() + batch_end_idx};

      if (complex_packing()) {
        std::vector<std::complex<double>> complex_vals;
        real_vec_to_complex_vec(complex_vals, real_vals);
        m_ckks_encoder->encode(complex_vals, m_scale, plain);
      } else {
        m_ckks_encoder->encode(real_vals, m_scale, plain);
      }
      m_encryptor->encrypt(plain, ciphers[data_idx]);
    }
    NGRAPH_INFO << "Creating execute message";
    auto execute_message =
        TCPMessage(ngraph::he::MessageType::execute, ciphers);
    NGRAPH_INFO << "Sending execute message with " << parameter_size
                << " ciphertexts";
    write_message(std::move(execute_message));
  } else if (msg_type == ngraph::he::MessageType::result) {
    size_t result_count = message.count();
    size_t element_size = message.element_size();

    NGRAPH_INFO << "Client got " << result_count << " results ";

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

      std::vector<double> outputs;
      decode_to_real_vec(plain, outputs, complex_packing());
      m_results.insert(m_results.end(), outputs.begin(), outputs.end());
    }
    NGRAPH_INFO << "Results size " << m_results.size();

    close_connection();
  } else if (msg_type == ngraph::he::MessageType::none) {
    close_connection();
  } else if (msg_type == ngraph::he::MessageType::encryption_parameters) {
    std::stringstream param_stream;
    param_stream.write(message.data_ptr(), message.element_size());
    m_encryption_params = seal::EncryptionParameters::Load(param_stream);
    NGRAPH_INFO << "Loaded encryption parmeters";

    set_seal_context();

    // Send public key
    std::stringstream pk_stream;
    m_public_key->save(pk_stream);
    auto pk_message = TCPMessage(ngraph::he::MessageType::public_key, 1,
                                 std::move(pk_stream));
    NGRAPH_INFO << "Sending public key";
    write_message(std::move(pk_message));

    // Send evaluation key
    std::stringstream evk_stream;
    m_relin_keys->save(evk_stream);
    auto evk_message =
        TCPMessage(ngraph::he::MessageType::eval_key, 1, std::move(evk_stream));
    NGRAPH_INFO << "Sending evaluation key";
    write_message(std::move(evk_message));
  } else if (msg_type == ngraph::he::MessageType::relu_request) {
    NGRAPH_INFO << "Received Relu request";
    auto relu = [](double d) { return d > 0 ? d : 0; };
    auto relu6 = [](double d) {
      if (d < 0) {
        return 0.0;
      }
      if (d > 6) {
        return 6.0;
      }
      return d;
    };
    size_t result_count = message.count();
    size_t element_size = message.element_size();
    NGRAPH_INFO << "Received Relu request with " << result_count << " elements"
                << " of size " << element_size;

    // TODO: reserve for efficiency
    std::stringstream post_relu_stream;
    std::vector<seal::Ciphertext> post_relu_ciphers(result_count);
#pragma omp parallel for
    for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
      seal::Ciphertext pre_relu_cipher;
      seal::Plaintext relu_plain;

      // Load cipher from stream
      std::stringstream pre_relu_cipher_stream;
      pre_relu_cipher_stream.write(
          message.data_ptr() + result_idx * element_size, element_size);
      pre_relu_cipher.load(m_context, pre_relu_cipher_stream);

      // Decrypt cipher
      m_decryptor->decrypt(pre_relu_cipher, relu_plain);

      std::vector<double> relu_vals;
      decode_to_real_vec(relu_plain, relu_vals, complex_packing());

      // Perform relu6
      // TODO: do relu instead of relu 6
      std::transform(relu_vals.begin(), relu_vals.end(), relu_vals.begin(),
                     relu6);

      if (complex_packing()) {
        std::vector<std::complex<double>> complex_relu_vals;
        real_vec_to_complex_vec(complex_relu_vals, relu_vals);
        m_ckks_encoder->encode(complex_relu_vals, m_scale, relu_plain);
      } else {
        m_ckks_encoder->encode(relu_vals, m_scale, relu_plain);
      }
      m_encryptor->encrypt(relu_plain, post_relu_ciphers[result_idx]);
    }
    NGRAPH_INFO << "Performed relu, saving ciphers to stream ";
    for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
      post_relu_ciphers[result_idx].save(post_relu_stream);
    }
    NGRAPH_INFO << "Writing relu_result message with " << result_count
                << " ciphertexts";

    auto relu_result_msg =
        TCPMessage(ngraph::he::MessageType::relu_result, result_count,
                   std::move(post_relu_stream));
    write_message(std::move(relu_result_msg));
  } else if (msg_type == ngraph::he::MessageType::max_request) {
    size_t complex_scale_factor = complex_packing() ? 2 : 1;
    size_t cipher_count = message.count();
    size_t element_size = message.element_size();

    std::vector<std::vector<double>> input_cipher_values(
        m_batch_size * complex_scale_factor,
        std::vector<double>(cipher_count, 0));

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
      std::vector<double> pre_max_value;
      decode_to_real_vec(pre_sort_plain, pre_max_value, complex_packing());

      for (size_t batch_idx = 0;
           batch_idx < m_batch_size * complex_scale_factor; ++batch_idx) {
        input_cipher_values[batch_idx][cipher_idx] = pre_max_value[batch_idx];
      }
    }

    // Get max of eachstd::vector of values
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
      assert(max_values.size() % 2 == 0);
      std::vector<std::complex<double>> max_complex_vals;
      real_vec_to_complex_vec(max_complex_vals, max_values);
      m_ckks_encoder->encode(max_complex_vals, m_scale, plain_max);
    } else {
      m_ckks_encoder->encode(max_values, m_scale, plain_max);
    }
    m_encryptor->encrypt(plain_max, cipher_max);
    cipher_max.save(max_stream);
    auto max_result_msg = TCPMessage(ngraph::he::MessageType::max_result, 1,
                                     std::move(max_stream));
    write_message(std::move(max_result_msg));
  } else if (msg_type == ngraph::he::MessageType::minimum_request) {
    // Stores (c_1a, c_1b, c_2a, c_b, ..., c_na, c_nb)
    // prints messsge (min(c_1a, c_1b), min(c_2a, c_2b), ..., min(c_na, c_nb))
    size_t cipher_count = message.count();
    assert(cipher_count % 2 == 0);
    size_t element_size = message.element_size();

    std::vector<std::vector<double>> input_cipher_values(
        cipher_count, std::vector<double>(m_batch_size, 0));

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
            std::min(input_cipher_values[cipher_idx][batch_idx],
                     input_cipher_values[cipher_idx + 1][batch_idx]);
      }
      // Encrypt minimum values
      seal::Ciphertext cipher_minimum;
      seal::Plaintext plain_minimum;
      m_ckks_encoder->encode(min_values, m_scale, plain_minimum);
      m_encryptor->encrypt(plain_minimum, cipher_minimum);
      cipher_minimum.save(minimum_stream);
    }

    auto minimum_result_msg =
        TCPMessage(ngraph::he::MessageType::minimum_result, cipher_count / 2,
                   std::move(minimum_stream));
    write_message(std::move(minimum_result_msg));
  } else {
    NGRAPH_INFO << "Unsupported message type: "
                << message_type_to_string(msg_type).c_str();
  }
}

void ngraph::he::HESealClient::close_connection() {
  NGRAPH_INFO << "Closing connection";
  m_tcp_client->close();
  m_is_done = true;
}
