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
#include <iostream>
#include <memory>
#include <string>
#include "tcp/tcp_message.hpp"

namespace ngraph {
namespace runtime {
namespace he {
class HESealClient {
 public:
  HESealClient(std::string hostname, std::size_t port) {}

 private:
  TCPClient m_tcp_client;
  seal::PublicKey m_public_key;
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
