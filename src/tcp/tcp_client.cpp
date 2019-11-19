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

#include "tcp/tcp_client.hpp"

#include <chrono>
#include <deque>
#include <memory>
#include <string>
#include <thread>

#include "boost/asio.hpp"
#include "logging/ngraph_he_log.hpp"
#include "ngraph/check.hpp"
#include "tcp/tcp_message.hpp"

namespace ngraph::runtime::he {
/// \brief Class representing a Client over a TCP connection

TCPClient::TCPClient(
    boost::asio::io_context& io_context,
    const boost::asio::ip::tcp::resolver::results_type& endpoints,
    const std::function<void(const TCPMessage&)>& message_handler)
    : m_io_context(io_context),
      m_socket(io_context),
      m_message_callback(std::bind(message_handler, std::placeholders::_1)) {
  do_connect(endpoints);
}

/// \brief Closes the socket
void TCPClient::close() {
  NGRAPH_HE_LOG(1) << "Closing socket";
  m_socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both);
  boost::asio::post(m_io_context, [this]() { m_socket.close(); });
}

/// \brief Asynchronously writes the message
/// \param[in,out] message Message to write
void TCPClient::write_message(TCPMessage&& message) {
  bool write_in_progress = !m_message_queue.empty();
  m_message_queue.push_back(std::move(message));
  if (!write_in_progress) {
    boost::asio::post(m_io_context, [this]() { do_write(); });
  }
}

void TCPClient::do_connect(
    const boost::asio::ip::tcp::resolver::results_type& endpoints,
    size_t delay_ms) {
  boost::asio::async_connect(
      m_socket, endpoints,
      [this, delay_ms, &endpoints](
          const boost::system::error_code& ec,
          const boost::asio::ip::tcp::endpoint& connect_endpoint) {
        static_cast<void>(connect_endpoint);  // Avoid unused-parameter warning
        if (!ec) {
          NGRAPH_HE_LOG(1) << "Connected to server";
          do_read_header();
        } else {
          NGRAPH_INFO << "error connecting to server: " << ec.message();
          std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
          size_t new_delay_ms = delay_ms;
          if (new_delay_ms < 1000) {
            new_delay_ms *= 2;
          }
          NGRAPH_INFO << "Trying to connect again";
          do_connect(endpoints, new_delay_ms);
        }
      });
}

void TCPClient::do_read_header() {
  if (m_read_buffer.size() < header_length) {
    m_read_buffer.resize(header_length);
  }
  boost::asio::async_read(
      m_socket, boost::asio::buffer(&m_read_buffer[0], header_length),
      [this](boost::system::error_code ec, std::size_t /* length */) {
        NGRAPH_CHECK(!ec || ec.message() == s_expected_teardown_message,
                     "Client error reading message header: ", ec.message());
        if (!ec) {
          size_t msg_len = TCPMessage::decode_header(m_read_buffer);
          do_read_body(msg_len);
        }
      });
}

void TCPClient::do_read_body(size_t body_length) {
  m_read_buffer.resize(header_length + body_length);
  boost::asio::async_read(
      m_socket, boost::asio::buffer(&m_read_buffer[header_length], body_length),
      [this](boost::system::error_code ec, std::size_t /* length */) {
        NGRAPH_CHECK(!ec || ec.message() == s_expected_teardown_message,
                     "Client error reading message body: ", ec.message());
        if (!ec) {
          m_read_message.unpack(m_read_buffer);
          m_message_callback(m_read_message);
          do_read_header();
        }
      });
}

void TCPClient::do_write() {
  auto message = m_message_queue.front();
  message.pack(m_write_buffer);
  NGRAPH_HE_LOG(4) << "Client writing message size " << m_write_buffer.size()
                   << " bytes";

  boost::asio::async_write(
      m_socket, boost::asio::buffer(m_write_buffer),
      [this](boost::system::error_code ec, std::size_t /* length */) {
        if (!ec) {
          m_message_queue.pop_front();
          if (!m_message_queue.empty()) {
            do_write();
          }
        } else {
          NGRAPH_ERR << "Client error writing message: " << ec.message();
        }
      });
}

}  // namespace ngraph::runtime::he
