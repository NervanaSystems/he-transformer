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
#include <functional>
#include <iostream>
#include <memory>
#include "tcp/tcp_message.hpp"
#include "tcp/tcp_session.hpp"

using boost::asio::ip::tcp;

namespace ngraph {
namespace runtime {
namespace he {
class TCPServer {
 public:
  TCPServer(boost::asio::io_context& io_context, const tcp::endpoint& endpoint,
            std::function<void(const runtime::he::TCPMessage&)> message_handler)
      : m_acceptor(io_context, endpoint),
        m_message_callback(std::bind(message_handler, std::placeholders::_1)) {
    accept_connection();
  }

 private:
  void accept_connection() {
    std::cout << "Server accepting connections" << std::endl;
    m_acceptor.async_accept(
        [this](boost::system::error_code ec, tcp::socket socket) {
          if (!ec) {
            std::cout << "Connection accepted" << std::endl;
            std::make_shared<TCPSession>(std::move(socket), m_message_callback)
                ->start();
          } else {
            std::cout << "error " << ec.message() << std::endl;
          }
          accept_connection();
        });
  }

  tcp::acceptor m_acceptor;
  std::function<void(const runtime::he::TCPMessage&)> m_message_callback;
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
