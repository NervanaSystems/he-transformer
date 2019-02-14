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

#pragma once

#include <boost/asio.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "he_seal_util.hpp"
#include "seal/context.h"
#include "seal/seal.h"
#include "tcp/tcp_client.hpp"
#include "tcp/tcp_message.hpp"

namespace ngraph {
namespace runtime {
namespace he {
class HESealClient {
 public:
  HESealClient(boost::asio::io_context& io_context,
               const tcp::resolver::results_type& endpoints,
               std::vector<float> inputs)
      : m_inputs{inputs} {
    auto client_callback = [this](const runtime::he::TCPMessage& message) {
      return handle_message(message);
    };

    seal::EncryptionParameters parms(
        seal::scheme_type::CKKS);  // TODO: enable BFV
    parms.set_poly_modulus_degree(1024);
    parms.set_coeff_modulus(
        {seal::util::global_variables::small_mods_30bit.begin(),
         seal::util::global_variables::small_mods_30bit.begin() + 4});

    m_context = seal::SEALContext::Create(parms);

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

    std::stringstream stream;
    m_public_key->save(stream);
    const std::string& pk_str = stream.str();
    const char* pk_cstr = pk_str.c_str();

    auto first_message = runtime::he::TCPMessage(MessageType::public_key, 1,
                                                 pk_str.size(), pk_cstr);

    m_tcp_client = std::make_shared<runtime::he::TCPClient>(
        io_context, endpoints, first_message, client_callback);

    m_thread = std::thread([&io_context]() { io_context.run(); });
  }

  ~HESealClient() {}

  const runtime::he::TCPMessage handle_message(
      const runtime::he::TCPMessage& message) {
    MessageType msg_type = message.message_type();

    std::cout << "Client got message: "
              << message_type_to_string(msg_type).c_str() << std::endl;

    if (msg_type == MessageType::public_key_request) {
    } else if (msg_type == MessageType::public_key_ack) {
      auto return_message = TCPMessage(MessageType::parameter_shape_request);

      return return_message;
    } else if (msg_type == MessageType::parameter_shape) {
      std::vector<size_t> shape(message.count());

      std::cout << "element size " << message.element_size() << std::endl;
      std::cout << "element count " << message.count() << std::endl;

      std::memcpy(shape.data(), message.data_ptr(), message.data_size());

      std::cout << "Shape " << join(shape, "x") << std::endl;

      auto shape_size = std::accumulate(begin(shape), end(shape), 1,
                                        std::multiplies<size_t>());

      std::cout << "shape size " << shape_size << std::endl;

      std::vector<seal::Ciphertext> ciphers;

      assert(m_inputs.size() == shape_size);

      std::stringstream cipher_stream;

      for (size_t i = 0; i < shape_size; ++i) {
        seal::Plaintext plain;
        m_ckks_encoder->encode(m_inputs[i], m_scale, plain);
        seal::Ciphertext c;
        m_encryptor->encrypt(plain, c);
        c.save(cipher_stream);
      }

      // c.save(cipher_stream);
      const std::string& cipher_str = cipher_stream.str();
      const char* cipher_cstr = cipher_str.c_str();

      size_t cipher_size = cipher_str.size();

      std::cout << "Cipher size " << cipher_size << std::endl;

      auto return_message = TCPMessage(MessageType::execute, shape_size,
                                       cipher_size, cipher_cstr);

      return return_message;
    } else if (msg_type == MessageType::result) {
      size_t count = message.count();
      size_t element_size = message.element_size();

      std::vector<seal::Ciphertext> result;

      for (size_t i = 0; i < count; ++i) {
        seal::Ciphertext cipher;
        std::stringstream cipher_stream;

        cipher_stream.write(message.data_ptr() + i * element_size,
                            element_size);

        cipher.load(m_context, cipher_stream);
        result.push_back(cipher);
        seal::Plaintext plain;
        m_decryptor->decrypt(cipher, plain);
        std::vector<double> output;
        m_ckks_encoder->decode(plain, output);

        std::cout << "output " << output[0] << std::endl;
        m_results.push_back((float)output[0]);
      }

      // sleep(10);  // Wait

      // close_connection();
      // std::cout << "Returning empty TCP message" << std::endl;
      return TCPMessage(MessageType::none);

    } else if (msg_type == MessageType::none) {
      close_connection();
    } else {
      std::cout << "Returning empty TCP message" << std::endl;
      return TCPMessage();
    }
  }

  void write_message(const runtime::he::TCPMessage& message) {
    m_tcp_client->write_message(message);
  }

  bool is_done() { return m_is_done; }

  std::vector<float> get_results() { return m_results; }

  void close_connection() {
    std::cout << "Closing connectiong" << std::endl;

    m_tcp_client->close();
    m_thread.detach();
    m_is_done = true;
  }

 private:
  std::shared_ptr<TCPClient> m_tcp_client;
  std::shared_ptr<seal::PublicKey> m_public_key;
  std::shared_ptr<seal::SecretKey> m_secret_key;
  std::shared_ptr<seal::SEALContext> m_context;
  std::shared_ptr<seal::Encryptor> m_encryptor;
  std::shared_ptr<seal::CKKSEncoder> m_ckks_encoder;
  std::shared_ptr<seal::Decryptor> m_decryptor;
  std::shared_ptr<seal::Evaluator> m_evaluator;
  std::shared_ptr<seal::KeyGenerator> m_keygen;
  std::shared_ptr<seal::RelinKeys> m_relin_keys;
  std::thread m_thread;
  double m_scale;
  bool m_is_done{false};
  std::vector<float> m_inputs;   // Function inputs
  std::vector<float> m_results;  // Function outputs

};  // namespace he
}  // namespace he
}  // namespace runtime
}  // namespace ngraph