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
  // Connects client to hostname:port and reads message
  // message_handler will handle responses from the server
  TCPClient(boost::asio::io_context& io_context,
            const tcp::resolver::results_type& endpoints,
            std::function<void(const runtime::he::TCPMessage&)> message_handler)
      : m_io_context(io_context),
        m_socket(io_context),
        m_message_callback(std::bind(message_handler, std::placeholders::_1)) {
    std::cout << "Client starting async connection" << std::endl;
    do_connect(endpoints);
  }

  void close() {
    std::cout << "Closing socket" << std::endl;
    m_socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both);
    boost::asio::post(m_io_context, [this]() { m_socket.close(); });
  }

  void write_message(const runtime::he::TCPMessage& message) {
    std::cout << "Client writing message" << std::endl;
    boost::asio::post(m_io_context, [this, message]() { do_write(message); });
  }

 private:
  void do_connect(const tcp::resolver::results_type& endpoints) {
    boost::asio::async_connect(
        m_socket, endpoints,
        [this](boost::system::error_code ec, tcp::endpoint) {
          if (!ec) {
            std::cout << "Connected to server" << std::endl;

            do_read_header();
            // do_write(m_read_message);
          } else {
            std::cout << "error connecting to server: " << ec.message()
                      << std::endl;
          }
        });
  }

  void do_read_header() {
    std::cout << "Client reading header " << std::endl;
    boost::asio::async_read(
        m_socket,
        boost::asio::buffer(m_read_message.header_ptr(),
                            runtime::he::TCPMessage::header_length),
        [this](boost::system::error_code ec, std::size_t length) {
          if (!ec && m_read_message.decode_header()) {
            do_read_body();
          } else {
            std::cout << "Client error reading header: " << ec.message()
                      << std::endl;
            m_socket.close();
            std::cout << "Closed socket" << std::endl;
          }
        });
  }

  void do_read_body() {
    std::cout << "Client reading body length " << std::endl;
    std::cout << "Body length " << m_read_message.body_length() << std::endl;
    boost::asio::async_read(
        m_socket,
        boost::asio::buffer(m_read_message.body_ptr(),
                            m_read_message.body_length()),
        [this](boost::system::error_code ec, std::size_t length) {
          if (!ec) {
            m_read_message.decode_body();
            std::cout << "Client read message length "
                      << m_read_message.num_bytes() << std::endl;

            m_message_callback(m_read_message);
            do_read_header();
          } else {
            std::cout << "Client error reading body; " << ec.message()
                      << std::endl;
            std::cout << "Closing socket" << std::endl;
            m_socket.close();
            std::cout << "Closed socket" << std::endl;
          }
        });
  }

  void do_write(const runtime::he::TCPMessage& message) {
    boost::asio::async_write(
        m_socket,
        boost::asio::buffer(message.header_ptr(), message.num_bytes()),
        [this](boost::system::error_code ec, std::size_t length) {
          if (!ec) {
            std::cout << "Client wrote message length " << length << std::endl;
            do_read_header();
          } else {
            std::cout << "Client error writing message: " << ec.message()
                      << std::endl;
          }
        });
  }

  boost::asio::io_context& m_io_context;
  TCPMessage m_read_message;
  tcp::socket m_socket;

  // How to handle the message
  std::function<void(const runtime::he::TCPMessage&)> m_message_callback;
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
