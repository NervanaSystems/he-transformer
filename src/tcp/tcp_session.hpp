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

#include "ngraph/log.hpp"
#include "tcp/tcp_message.hpp"

using boost::asio::ip::tcp;

namespace ngraph {
namespace he {
class TCPSession : public std::enable_shared_from_this<TCPSession> {
  using data_buffer = std::vector<char>;
  size_t header_length = ngraph::he::TCPMessage::header_length;

 public:
  TCPSession(tcp::socket socket,
             std::function<void(const ngraph::he::TCPMessage&)> message_handler)
      : m_socket(std::move(socket)),
        m_writing(false),
        m_message_callback(std::bind(message_handler, std::placeholders::_1)) {}

  void start() { do_read_header(); }

 public:
  void do_read_header() {
    NGRAPH_INFO << "server do_read_header";
    m_read_buffer.resize(header_length);
    auto self(shared_from_this());
    boost::asio::async_read(
        m_socket, boost::asio::buffer(m_read_buffer),
        [this, self](boost::system::error_code ec, std::size_t length) {
          if (!ec) {
            size_t msg_len = m_read_message.decode_header(m_read_buffer);
            NGRAPH_INFO << "server read hader for msg len " << msg_len;
            do_read_body(msg_len);
          } else {
            if (ec) {
              // End of file is expected on teardown
              if (ec.message() != "End of file") {
                NGRAPH_INFO << "Server error reading body: " << ec.message();
              }
            }
          }
        });
  }

  void do_read_body(size_t body_length = 0) {
    NGRAPH_INFO << "server reading body size " << body_length;
    m_read_buffer.resize(header_length + body_length);

    auto self(shared_from_this());
    boost::asio::async_read(
        m_socket,
        boost::asio::buffer(&m_read_buffer[header_length], body_length),
        [this, self](boost::system::error_code ec, std::size_t length) {
          if (!ec) {
            m_read_message.unpack(m_read_buffer);
            m_message_callback(m_read_message);
            do_read_header();
          } else {
            NGRAPH_INFO << "Server error reading message: " << ec.message();
            throw std::runtime_error("Server error reading message");
          }
        });
  }

  void write_message(ngraph::he::TCPMessage&& message) {
    NGRAPH_INFO << "server write  message";
    bool write_in_progress = is_writing();
    m_message_queue.emplace_back(std::move(message));
    if (!write_in_progress) {
      do_write();
    }
  }

  bool is_writing() const { return !m_message_queue.empty(); }

  std::condition_variable& is_writing_cond() { return m_is_writing; }

 private:
  void do_write() {
    NGRAPH_INFO << "server do_write";
    std::lock_guard<std::mutex> lock(m_write_mtx);
    m_is_writing.notify_all();
    auto self(shared_from_this());

    NGRAPH_INFO << "m_message_queue.size() " << m_message_queue.size();

    auto message = m_message_queue.front();
    data_buffer write_buf;
    message.pack(write_buf);

    NGRAPH_INFO << "Buffer size " << write_buf.size();

    boost::asio::write(m_socket, boost::asio::buffer(write_buf));
    m_message_queue.pop_front();

    if (!m_message_queue.empty()) {
      NGRAPH_INFO << "Message queue not empty; do_write()";
      do_write();
    } else {
      NGRAPH_INFO << "Notifying done writing";
      m_is_writing.notify_all();
    }
  }

 private:
  std::deque<ngraph::he::TCPMessage> m_message_queue;
  data_buffer m_read_buffer;
  TCPMessage m_read_message;
  tcp::socket m_socket;
  bool m_writing;
  std::condition_variable m_is_writing;
  std::mutex m_write_mtx;

  // Called after message is received
  std::function<void(const ngraph::he::TCPMessage&)> m_message_callback;
};
}  // namespace he
}  // namespace ngraph
