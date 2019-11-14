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

#include <deque>
#include <memory>
#include <string>

#include "boost/asio.hpp"
#include "logging/ngraph_he_log.hpp"
#include "tcp/tcp_message.hpp"

namespace ngraph::runtime::he {
/// \brief Class representing a Client over a TCP connection
class TCPClient {
 public:
  using data_buffer = TCPMessage::data_buffer;
  size_t header_length = TCPMessage::header_length;

  /// \brief Connects client to hostname:port and reads message
  /// \param[in] io_context Boost context for I/O functionality
  /// \param[in] endpoints Socket to connect to
  /// \param[in] message_handler Function to handle responses from the server
  TCPClient(boost::asio::io_context& io_context,
            const boost::asio::ip::tcp::resolver::results_type& endpoints,
            std::function<void(const TCPMessage&)> message_handler);

  /// \brief Closes the socket
  void close();

  /// \brief Asynchronously writes the message
  /// \param[in,out] message Message to write
  void write_message(const TCPMessage&& message);

 private:
  void do_connect(const boost::asio::ip::tcp::resolver::results_type& endpoints,
                  size_t delay_ms = 10);

  void do_read_header();

  void do_read_body(size_t body_length);

  void do_write();

  boost::asio::io_context& m_io_context;
  boost::asio::ip::tcp::socket m_socket;

  data_buffer m_read_buffer;
  data_buffer m_write_buffer;
  TCPMessage m_read_message;
  std::deque<TCPMessage> m_message_queue;

  inline static std::string s_expected_teardown_message{"End of file"};

  std::function<void(const TCPMessage&)> m_message_callback;
};
}  // namespace ngraph::runtime::he
