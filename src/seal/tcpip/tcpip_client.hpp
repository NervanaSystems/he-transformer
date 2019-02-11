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
#include <string>

using boost::asio::ip::tcp;

namespace ngraph {
namespace runtime {
namespace he {
class TCPClient {
  // Connects client to hostname:port
  TCPClient(const std::string& hostname, const size_t port) {
    boost::asio::io_service io_service;
    tcp::resolver resolver(io_service);
    tcp::resolver::query query(hostname, port);
    tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);

    tcp::socket socket(io_service);

    boost::system::error_code error;
    socket.connect(endpoint_iterator, error);

    if (error) {
      throw boost::system::system_error(error);
    }
  }
};

}  // namespace he
}  // namespace runtime
}  // namespace ngraph
