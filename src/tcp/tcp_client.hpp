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
  // Connects client to hostname:port and reads message
  // message_handler will handle responses from the server
  TCPClient(
      boost::asio::io_context& io_context,
      const tcp::resolver::results_type& endpoints,
      std::function<void(const ngraph::he::NewTCPMessage&)> message_handler)
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

  void write_message(ngraph::he::TCPMessage&& message) {
    bool write_in_progress = !m_message_queue.empty();
    m_message_queue.emplace_back(std::move(message));
    if (!write_in_progress) {
      boost::asio::post(m_io_context, [this]() { do_write(); });
    }
  }

  // TODO: send copy or rvalue
  void write_message(ngraph::he::NewTCPMessage message) {
    bool write_in_progress = !m_new_message_queue.empty();
    m_new_message_queue.push_back(message);
    if (!write_in_progress) {
      boost::asio::post(m_io_context, [this]() { do_new_write(); });
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
    NGRAPH_INFO << "Client do read header";
    NGRAPH_INFO << "ngraph::he::NewTCPMessage::header_length "
                << ngraph::he::NewTCPMessage::header_length;
    boost::asio::async_read(
        m_socket,
        boost::asio::buffer(m_new_read_message.size_ptr(),
                            ngraph::he::NewTCPMessage::header_length),
        [this](boost::system::error_code ec, std::size_t length) {
          if (!ec && m_new_read_message.decode_header()) {
            NGRAPH_INFO << "Client read header size " << length << " bits";
            do_read_body();
          } else {
            // End of file is expected on teardown
            if (ec.message() != "End of file") {
              NGRAPH_INFO << "Client error reading header: " << ec.message();
            }
          }
        });
  }

  void do_read_body() {
    NGRAPH_INFO << "Client do read body";
    boost::asio::streambuf receive_streambuf;
    boost::asio::async_read(
        m_socket, receive_streambuf,
        [this, &receive_streambuf](boost::system::error_code ec,
                                   std::size_t length) {
          if (!ec) {
            m_new_read_message.read_from_buffer(receive_streambuf);
            m_message_callback(m_new_read_message);
            do_read_header();
          } else {
            // End of file is expected on teardown
            if (ec.message() != "End of file") {
              NGRAPH_INFO << "Client error reading body: " << ec.message();
            }
          }
        });
  }

  void do_write() {
    NGRAPH_INFO << "Client do_write";
    boost::asio::async_write(
        m_socket,
        boost::asio::buffer(m_message_queue.front().header_ptr(),
                            m_message_queue.front().num_bytes()),
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

  void do_new_write() {
    NGRAPH_INFO << "Client do_new_write";
    boost::asio::streambuf send_streambuf;
    m_new_message_queue.front().write_to_buffer(send_streambuf);

    boost::asio::async_write(
        m_socket, send_streambuf,
        [this](boost::system::error_code ec, std::size_t length) {
          if (!ec) {
            m_new_message_queue.pop_front();
            if (!m_new_message_queue.empty()) {
              do_new_write();
            }
          } else {
            NGRAPH_INFO << "Client error writing message: " << ec.message();
          }
        });
  }

  boost::asio::io_context& m_io_context;
  tcp::socket m_socket;

  NewTCPMessage m_new_read_message;
  TCPMessage m_read_message;
  std::deque<ngraph::he::TCPMessage> m_message_queue;
  std::deque<ngraph::he::NewTCPMessage> m_new_message_queue;

  bool m_first_connect;

  // How to handle the message
  std::function<void(const ngraph::he::NewTCPMessage&)> m_message_callback;
};
}  // namespace he
}  // namespace ngraph
