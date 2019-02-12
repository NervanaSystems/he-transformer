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
#include "tcp/tcp_message.hpp"

using boost::asio::ip::tcp;

namespace ngraph {
namespace runtime {
namespace he {
class TCPClient {
 public:
  // Connects client to hostname:port
  TCPClient(boost::asio::io_context& io_context,
            const tcp::resolver::results_type& endpoints)
      : m_io_context(io_context), m_socket(io_context) {
    std::cout << "Client starting async connection" << std::endl;
    do_connect(endpoints);
  }

  ~TCPClient() { close(); }

  void write_message(const runtime::he::TCPMessage& message) {
    std::cout << "Writing message" << std::endl;
    boost::asio::async_write(
        m_socket, boost::asio::buffer(message.data(), message.size()),
        [this](boost::system::error_code ec, std::size_t length) {
          if (!ec) {
            std::cout << "Wrote message length " << length << std::endl;
          } else {
            std::cout << "error writing message: " << ec << std::endl;
          }
        });
  }

 private:
  void do_connect(const tcp::resolver::results_type& endpoints) {
    boost::asio::async_connect(
        m_socket, endpoints,
        [this](boost::system::error_code ec, tcp::endpoint) {
          if (!ec) {
            std::cout << "Connected to server" << std::endl;
          } else {
            std::cout << "error connecting to server: " << ec << std::endl;
          }
        });
  }

  void close() {
    std::cout << "Closing socket" << std::endl;
    boost::asio::post(m_io_context, [this]() { m_socket.close(); });
  }

  boost::asio::io_context& m_io_context;
  tcp::socket m_socket;
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
