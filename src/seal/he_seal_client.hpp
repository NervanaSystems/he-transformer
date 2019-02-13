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
               const tcp::resolver::results_type& endpoints) {
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

    m_thread = std::thread([&io_context]() { io_context.run(); });

    auto m_context_data = m_context->context_data();
    m_scale = static_cast<double>(
        m_context_data->parms().coeff_modulus().back().value());

    std::stringstream stream;
    m_public_key->save(stream);
    const std::string& pk_str = stream.str();
    const char* pk_cstr = pk_str.c_str();
    NGRAPH_INFO << "Size of pk " << pk_str.size();

    auto first_message = runtime::he::TCPMessage(MessageType::public_key, 1,
                                                 pk_str.size(), pk_cstr);

    m_tcp_client = std::make_shared<runtime::he::TCPClient>(
        io_context, endpoints, first_message, client_callback);
  }

  const runtime::he::TCPMessage handle_message(
      const runtime::he::TCPMessage& message) {
    std::cout << "HESealClient callback for message" << std::endl;

    MessageType msg_type = message.message_type();

    if (msg_type == MessageType::public_key_request) {
      std::cout << "Got message public_key_request" << std::endl;
    } else if (msg_type == MessageType::public_key_ack) {
      std::cout << "Got message public_key_ack" << std::endl;

      /* size_t pk_size = message.data_size();
      std::stringstream pk_stream;
      pk_stream.write(message.data_ptr(), pk_size);
      m_public_key.load(m_context, pk_stream);
      assert(m_public_key.is_valid_for(m_context));

      std::cout << "Copied public key from server" << std::endl; */

      // m_encryptor = std::make_shared<seal::Encryptor>(m_context,
      // m_public_key);

      std::vector<seal::Ciphertext> ciphers;

      std::vector<double> input{1.1};
      seal::Plaintext plain;
      m_ckks_encoder->encode(input, m_scale, plain);
      seal::Ciphertext c;
      m_encryptor->encrypt(plain, c);

      std::cout << "m_scale " << m_scale << std::endl;
      print_seal_context(*m_context);

      std::stringstream cipher_stream;
      c.save(cipher_stream);
      const std::string& cipher_str = cipher_stream.str();
      const char* cipher_cstr = cipher_str.c_str();

      size_t cipher_size = cipher_str.size();

      auto return_message =
          TCPMessage(MessageType::inference, 1, cipher_size, cipher_cstr);

      std::cout << "Returning ciphertext " << std::endl;
      return return_message;
    } else if (msg_type == MessageType::result) {
      std::cout << "Cleint got result type" << std::endl;
      throw std::domain_error("So far so good in client");
    }

    else if (msg_type == MessageType::public_key_ack) {
    }

    else {
      std::cout << "Returning empty TCP message" << std::endl;
      return TCPMessage();
    }
  }

  void write_message(const runtime::he::TCPMessage& message) {
    std::cout << "HESealClient client writing tcp message" << std::endl;
    m_tcp_client->write_message(message);
  }

  void close_connection() {
    m_tcp_client->close();
    m_thread.join();
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
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
