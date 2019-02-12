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
#include <memory>
#include "tcp/tcp_message.hpp"

using boost::asio::ip::tcp;

namespace ngraph {
namespace runtime {
namespace he {
class TCPSession : public std::enable_shared_from_this<TCPSession> {
 public:
  TCPSession(tcp::socket socket) : m_socket(std::move(socket)) {}

  void start() { do_read_header(); }

 private:
  void do_read_header() {
    auto self(shared_from_this());
    boost::asio::async_read(
        m_socket,
        boost::asio::buffer(m_message.data(),
                            runtime::he::TCPMessage::header_length),
        [this, self](boost::system::error_code ec, std::size_t /*length*/) {
          if (!ec && m_message.decode_header()) {
            do_read_body();
          } else {
            m_socket.close();
          }
        });
  }

  void do_read_body() { std::cout << "Reading message body" << std::endl; }

  void read_message(const runtime::he::TCPMessage& message) {
    boost::asio::async_write(
        m_socket, boost::asio::buffer(message.data(), message.size()),
        [this](boost::system::error_code ec, std::size_t length) {
          if (!ec) {
            std::cout << "Wrote message length " << length << std::endl;
          } else {
            std::cout << "error" << ec << std::endl;
          }
        });
  }

  TCPMessage m_message;
  tcp::socket m_socket;
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
