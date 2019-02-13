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
#include "seal/context.h"
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
    size_t N = 100;
    void* x = malloc(N);
    memset(x, 0, N);
    assert(x != nullptr);
    auto first_message = runtime::he::TCPMessage(
        runtime::he::MessageType::public_key_request, 0, N, (char*)x);

    m_tcp_client = std::make_shared<runtime::he::TCPClient>(
        io_context, endpoints, first_message, client_callback);

    m_thread = std::thread([&io_context]() { io_context.run(); });
  }

  const runtime::he::TCPMessage& handle_message(
      const runtime::he::TCPMessage& message) {
    std::cout << "HESealClient callback for message" << std::endl;

    MessageType msg_type = message.get_message_type();

    if (msg_type == MessageType::public_key_request) {
      std::cout << "Got message public_key_request" << std::endl;
    } else if (msg_type == MessageType::public_key) {
      std::cout << "Got message public_key" << std::endl;
    }

    return TCPMessage();
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
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
