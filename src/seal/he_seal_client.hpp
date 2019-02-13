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
    auto first_message =
        runtime::he::TCPMessage(runtime::he::MessageType::public_key_request);

    m_tcp_client = std::make_shared<runtime::he::TCPClient>(
        io_context, endpoints, first_message, client_callback);

    seal::EncryptionParameters parms(
        seal::scheme_type::CKKS);  // TODO: enable BFV
    parms.set_poly_modulus_degree(1024);
    parms.set_coeff_modulus(
        {seal::util::global_variables::small_mods_30bit.begin(),
         seal::util::global_variables::small_mods_30bit.begin() + 4});

    m_context = seal::SEALContext::Create(parms);
    m_encoder = std::make_shared<seal::CKKSEncoder>(m_context);
    m_thread = std::thread([&io_context]() { io_context.run(); });

    auto m_context_data = m_context->context_data();
    m_scale = static_cast<double>(
        m_context_data->parms().coeff_modulus().back().value());
  }

  const runtime::he::TCPMessage handle_message(
      const runtime::he::TCPMessage& message) {
    std::cout << "HESealClient callback for message" << std::endl;

    MessageType msg_type = message.message_type();

    if (msg_type == MessageType::public_key_request) {
      std::cout << "Got message public_key_request" << std::endl;
    } else if (msg_type == MessageType::public_key) {
      std::cout << "Got message public_key" << std::endl;

      size_t pk_size = message.data_size();
      std::cout << "pk_size " << pk_size << std::endl;

      std::stringstream pk_stream;

      pk_stream.write(message.data_ptr(), pk_size);

      std::cout << "Wrote pk to stringstream" << std::endl;

      m_public_key.load(m_context, pk_stream);

      std::cout << "Copied public key from server" << std::endl;

      m_encryptor = std::make_shared<seal::Encryptor>(m_context, m_public_key);

      std::vector<double> input{0.0, 1.1, 2.2, 3.3};

      seal::Plaintext plain;
      m_encoder->encode(input, m_scale, plain);
      seal::Ciphertext c;

      m_encryptor->encrypt(plain, c);

      std::cout << "m_scale " << m_scale << std::endl;

      std::stringstream cipher_stream;
      c.save(cipher_stream);
      const std::string& cipher_str = cipher_stream.str();
      const char* cipher_cstr = cipher_str.c_str();

      std::cout << "Ciphertext size " << sizeof(seal::Ciphertext) << std::endl;

      auto return_message = TCPMessage(MessageType::inference, 1,
                                       sizeof(seal::Ciphertext), cipher_cstr);

      std::cout << "Returning ciphertext " << std::endl;
      return return_message;
    } else {
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
  seal::PublicKey m_public_key;
  std::shared_ptr<seal::SEALContext> m_context;
  std::thread m_thread;
  std::shared_ptr<seal::Encryptor> m_encryptor;
  std::shared_ptr<seal::CKKSEncoder> m_encoder;
  double m_scale;
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
