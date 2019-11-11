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
#include <deque>
#include <functional>
#include <memory>
#include <mutex>

#include "logging/ngraph_he_log.hpp"
#include "tcp/tcp_message.hpp"

namespace ngraph::he {
/// \brief Class representing a session over TCP
class TCPSession : public std::enable_shared_from_this<TCPSession> {
  using data_buffer = TCPMessage::data_buffer;
  size_t header_length = TCPMessage::header_length;

 public:
  /// \brief Constructs a session with a given message handler
  TCPSession(boost::asio::ip::tcp::socket socket,
             std::function<void(const TCPMessage&)> message_handler);

  /// \brief Start the session
  void start() { do_read_header(); }

  /// \brief Reads a header
  void do_read_header();

  /// \brief Reads message body of specified length
  /// \param[in] body_length Number of bytes to read
  void do_read_body(size_t body_length);

  /// \brief Adds a message to the message-writing queue
  /// \param[in,out] message Message to write
  void write_message(TCPMessage&& message);

  /// \brief Returns whether or not a message is queued to be written
  bool is_writing() const { return !m_message_queue.empty(); }

  /// \brief Returns a condition variable notified when the session is done
  /// writing a message
  std::condition_variable& is_writing_cond() { return m_is_writing; }

 private:
  void do_write();

 private:
  std::deque<TCPMessage> m_message_queue;
  TCPMessage m_read_message;

  data_buffer m_read_buffer;
  data_buffer m_write_buffer;
  boost::asio::ip::tcp::socket m_socket;
  std::condition_variable m_is_writing;
  std::mutex m_write_mtx;

  inline static std::string s_expected_teardown_message{"End of file"};

  std::function<void(const TCPMessage&)> m_message_callback;
};
}  // namespace ngraph::he
