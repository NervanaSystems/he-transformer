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
#include <chrono>
#include <deque>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "ngraph/log.hpp"

#include "tcp/tcp_message.hpp"

using boost::asio::ip::tcp;

namespace ngraph {
namespace he {
class TCPClient {
 public:
  using data_buffer = ngraph::he::TCPMessage::data_buffer;
  size_t header_length = ngraph::he::TCPMessage::header_length;

  // Connects client to hostname:port and reads message
  // message_handler will handle responses from the server
  TCPClient(boost::asio::io_context& io_context,
            const tcp::resolver::results_type& endpoints,
            std::function<void(const ngraph::he::TCPMessage&)> message_handler)
      : m_io_context(io_context),
        m_socket(io_context),
        m_first_connect(true),
        m_message_callback(std::bind(message_handler, std::placeholders::_1)) {
    do_connect(endpoints);
  }

  void close() {
    NGRAPH_INFO << "Closing socket";
    m_socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both);
    boost::asio::post(m_io_context, [this]() { m_socket.close(); });
  }

  void write_message(const ngraph::he::TCPMessage&& message) {
    bool write_in_progress = !m_message_queue.empty();
    m_message_queue.push_back(std::move(message));
    if (!write_in_progress) {
      boost::asio::post(m_io_context, [this]() { do_write(); });
    }
  }

 private:
  void do_connect(const tcp::resolver::results_type& endpoints,
                  size_t delay_ms = 10) {
    boost::asio::async_connect(
        m_socket, endpoints,
        [this, delay_ms, &endpoints](boost::system::error_code ec,
                                     tcp::endpoint) {
          if (!ec) {
            NGRAPH_INFO << "Connected to server";
            do_read_header();
          } else {
            if (true || m_first_connect) {
              NGRAPH_INFO << "error connecting to server: " << ec.message();
              m_first_connect = false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            size_t new_delay_ms = delay_ms;
            if (new_delay_ms < 1000) {
              new_delay_ms *= 2;
            }
            NGRAPH_INFO << "Trying to connect again";
            do_connect(std::move(endpoints), new_delay_ms);
          }
        });
  }

  void do_read_header() {
    m_read_buffer.resize(header_length);
    boost::asio::async_read(
        m_socket, boost::asio::buffer(m_read_buffer),
        [this](boost::system::error_code ec, std::size_t length) {
          if (!ec) {
            size_t msg_len = m_read_message.decode_header(m_read_buffer);
            do_read_body(msg_len);
          } else {
            if (ec.message() != s_expected_teardown_message.c_str()) {
              NGRAPH_INFO << "Client error reading header: " << ec.message();
            }
          }
        });
  }

  void do_read_body(size_t body_length) {
    m_read_buffer.resize(header_length + body_length);

    boost::asio::async_read(
        m_socket,
        boost::asio::buffer(&m_read_buffer[header_length], body_length),
        [this](boost::system::error_code ec, std::size_t length) {
          if (!ec) {
            m_read_message.unpack(m_read_buffer);
            m_message_callback(m_read_message);
            do_read_header();
          } else {
            if (ec.message() != s_expected_teardown_message.c_str()) {
              NGRAPH_INFO << "Client error reading body: " << ec.message();
            }
          }
        });
  }

  void do_write() {
    auto message = m_message_queue.front();
    message.pack(m_write_buffer);

    boost::asio::async_write(
        m_socket, boost::asio::buffer(m_write_buffer),
        [this](boost::system::error_code ec, std::size_t length) {
          if (!ec) {
            m_message_queue.pop_front();
            if (!m_message_queue.empty()) {
              do_write();
            }
          } else {
            NGRAPH_INFO << "Client error writing message: " << ec.message();
          }
        });
  }

  boost::asio::io_context& m_io_context;
  tcp::socket m_socket;

  data_buffer m_read_buffer;
  data_buffer m_write_buffer;
  TCPMessage m_read_message;
  std::deque<ngraph::he::TCPMessage> m_message_queue;

  inline static std::string s_expected_teardown_message{"End of file"};

  bool m_first_connect;
  std::function<void(const ngraph::he::TCPMessage&)> m_message_callback;
};
}  // namespace he
}  // namespace ngraph
