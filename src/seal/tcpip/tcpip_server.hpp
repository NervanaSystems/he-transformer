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

using boost::asio::ip::tcp;

namespace ngraph {
namespace runtime {
namespace he {
class TCPServer {
  TCPServer(const size_t port) : m_port(port) {
    boost::asio::io_service io_service;
    m_acceptor = acceptor(io_service, tcp::endpoint(tcp::v4(), port));
  }

  size_t m_port;
  tcp::acceptor m_acceptor;
};

}  // namespace he
}  // namespace runtime
}  // namespace ngraph
