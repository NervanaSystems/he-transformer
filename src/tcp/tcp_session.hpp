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
#include <memory>
#include <mutex>

#include "tcp/tcp_message.hpp"

using boost::asio::ip::tcp;

namespace ngraph {
namespace he {
class TCPSession : public std::enable_shared_from_this<TCPSession> {
 public:
  TCPSession(tcp::socket socket,
             std::function<void(const ngraph::he::TCPMessage&)> message_handler)
      : m_socket(std::move(socket)),
        m_message_callback(std::bind(message_handler, std::placeholders::_1)),
        m_writing(false) {}

  void start() { do_read_header(); }

  tcp::socket& socket() { return m_socket; }

 public:
  void do_read_header() {
    auto self(shared_from_this());
    boost::asio::async_read(
        m_socket,
        boost::asio::buffer(m_message.header_ptr(),
                            ngraph::he::TCPMessage::header_length),
        [this, self](boost::system::error_code ec, std::size_t length) {
          if (!ec & m_message.decode_header()) {
            do_read_body();
          } else {
            if (ec) {
              std::stringstream ss;
              ss << "Server error reading message: " << ec.message()
                 << std::endl;
              throw std::runtime_error(ss.str());
            }
          }
        });
  }

  void do_read_body() {
    auto self(shared_from_this());
    boost::asio::async_read(
        m_socket,
        boost::asio::buffer(m_message.body_ptr(), m_message.body_length()),
        [this, self](boost::system::error_code ec, std::size_t length) {
          if (!ec) {
            m_message.decode_body();
            m_message_callback(m_message);
            do_read_header();
          } else {
            std::stringstream ss;
            ss << "Error reading message body: " << ec.message() << std::endl;
            throw std::runtime_error(ss.str());
          }
        });
  }

  void do_write(const TCPMessage&& message) {
    std::lock_guard<std::mutex> lock(m_write_mtx);
    auto self(shared_from_this());
    m_writing = true;
    boost::asio::async_write(
        m_socket,
        boost::asio::buffer(message.header_ptr(), message.num_bytes()),
        [this, self](boost::system::error_code ec, std::size_t length) {
          if (ec) {
            std::cout << "Error writing message in session: " << ec.message()
                      << std::endl;
          } else {
            m_writing = false;
          }
        });
  }

  bool is_writing() const { return m_writing; }

  TCPMessage m_message;
  tcp::socket m_socket;
  bool m_writing;
  std::mutex m_write_mtx;

  // Called after message is received
  std::function<void(const ngraph::he::TCPMessage&)> m_message_callback;
};
}  // namespace he
}  // namespace ngraph
