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

#include "tcp/tcp_session.hpp"

#include <deque>
#include <functional>
#include <memory>
#include <mutex>

#include "boost/asio.hpp"
#include "logging/ngraph_he_log.hpp"
#include "ngraph/check.hpp"
#include "tcp/tcp_message.hpp"

namespace ngraph::runtime::he {
TCPSession::TCPSession(boost::asio::ip::tcp::socket socket,
                       std::function<void(const TCPMessage&)> message_handler)
    : m_socket(std::move(socket)),
      m_message_callback(std::bind(message_handler, std::placeholders::_1)) {}

void TCPSession::do_read_header() {
  if (m_read_buffer.size() < header_length) {
    m_read_buffer.resize(header_length);
  }
  auto self(shared_from_this());
  boost::asio::async_read(
      m_socket, boost::asio::buffer(&m_read_buffer[0], header_length),
      [this, self](boost::system::error_code ec, std::size_t length) {
        NGRAPH_CHECK(!ec || ec.message() == s_expected_teardown_message.c_str(),
                     "Server error reading message header: ", ec.message());
        if (!ec) {
          size_t msg_len = m_read_message.decode_header(m_read_buffer);
          do_read_body(msg_len);
        }
      });
}

void TCPSession::do_read_body(size_t body_length) {
  m_read_buffer.resize(header_length + body_length);

  auto self(shared_from_this());
  boost::asio::async_read(
      m_socket, boost::asio::buffer(&m_read_buffer[header_length], body_length),
      [this, self](boost::system::error_code ec, std::size_t length) {
        NGRAPH_CHECK(!ec || ec.message() == s_expected_teardown_message.c_str(),
                     "Server error reading message body: ", ec.message());
        if (!ec) {
          m_read_message.unpack(m_read_buffer);
          m_message_callback(m_read_message);
          do_read_header();
        }
      });
}

void TCPSession::write_message(TCPMessage&& message) {
  bool write_in_progress = is_writing();
  m_message_queue.emplace_back(std::move(message));
  if (!write_in_progress) {
    do_write();
  }
}

void TCPSession::do_write() {
  std::lock_guard<std::mutex> lock(m_write_mtx);
  m_is_writing.notify_all();
  auto self(shared_from_this());
  auto message = m_message_queue.front();
  message.pack(m_write_buffer);
  NGRAPH_HE_LOG(4) << "Server writing message size " << m_write_buffer.size()
                   << " bytes";

  boost::asio::async_write(
      m_socket, boost::asio::buffer(m_write_buffer),
      [this, self](boost::system::error_code ec, std::size_t length) {
        NGRAPH_CHECK(!ec, "Server error writing message: ", ec.message());
        m_message_queue.pop_front();
        if (!m_message_queue.empty()) {
          do_write();
        } else {
          m_is_writing.notify_all();
        }
      });
}

}  // namespace ngraph::runtime::he
