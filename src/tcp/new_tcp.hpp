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
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "ngraph/log.hpp"

#include "tcp/tcp_message.hpp"

using boost::asio::ip::tcp;

namespace ngraph {
namespace he {
class TestClient {
 public:
  TestClient(
      std::string hostname, size_t port,
      std::function<void(const std::string&, bool& exit)> message_handler)
      : m_stream(hostname, std::to_string(port)) {
    NGRAPH_INFO << "Created client";

    while (true) {
    }

    NGRAPH_INFO << "Closing client";
  }

  void write(const std::ostream& stream) { m_stream << stream; }

  void read_message() {
    bool exit{false};
    if (std::getline(m_stream, line)) {
      NGRAPH_INFO << "Client got message " << line;

      message_handler(line, exit);
      if (exit) {
        NGARPH_INFO << "Client got exit message";
        break;
      }
    } else {
      NGRAPH_INFO << "Error getting stream in client";
    }

    read_message();
  }

 private:
  tcp::iostream m_stream;
  std::string m_message;
};

class TestServer {
 public:
  TestServer(
      size_t port,
      std::function<void(const std::string&, bool& exit)> message_handler)
      : m_port(port) {
    tcp::resolver resolver(m_io_context);
    tcp::endpoint server_endpoints(tcp::v4(), m_port);
    m_acceptor =
        std::make_unique<tcp::acceptor>(m_io_context, server_endpoints);
    NGRAPH_INFO << "Created acceptor";

    boost::system::error_code ec;
    while (true) {
      m_acceptor->accept(m_stream.socket(), ec);

      if (ec) {
        NGRAPH_INFO << "error accepting connection " << ec.message()
                    << ". Trying again";
        continue;
      }

      NGRAPH_INFO << "Connection accepted";
      NGRAPH_INFO << "Session started";

      m_thread()
    }

    if (ec) {
      NGRAPH_INFO << "error accepting connection " << ec.message()
                  << ". Trying again";
      m_acceptor->accept(m_stream.socket(), ec);
    } else {
      m_stream << "test message";
      m_stream << std::endl;
      m_stream.flush();
    }
  }

 private:
  std::unique_ptr<tcp::acceptor> m_acceptor;
  boost::asio::io_context m_io_context;
  tcp::iostream m_stream;
  size_t m_port;
  std::thread m_thread;
};
}  // namespace he
}  // namespace ngraph
