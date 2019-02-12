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

#include <chrono>
#include "boost/asio.hpp"
#include "he_backend.hpp"
#include "ngraph/ngraph.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "tcp/tcp_client.hpp"
#include "tcp/tcp_message.hpp"
#include "tcp/tcp_server.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, tcp_client_server_init) {
  size_t port = 35000;

  boost::asio::io_context io_context;
  tcp::resolver resolver(io_context);
  auto client_endpoints = resolver.resolve("localhost", std::to_string(port));
  tcp::endpoint server_endpoints(tcp::v4(), port);

  auto server = runtime::he::TCPServer(io_context, server_endpoints);
  auto client = runtime::he::TCPClient(io_context, client_endpoints);

  io_context.run();
}

NGRAPH_TEST(${BACKEND_NAME}, tcp_client_server_init2) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_seal_ckks_backend =
      static_cast<runtime::he::he_seal::HESealCKKSBackend*>(backend.get());

  size_t port = he_seal_ckks_backend->get_port();
  NGRAPH_INFO << "Port " << port;

  boost::asio::io_context io_context;
  tcp::resolver resolver(io_context);
  auto client_endpoints = resolver.resolve("localhost", std::to_string(port));
  tcp::endpoint server_endpoints(tcp::v4(), port);

  auto client = runtime::he::TCPClient(io_context, client_endpoints);

  io_context.run();
}

NGRAPH_TEST(${BACKEND_NAME}, tcp_client_server_init3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_seal_ckks_backend =
      static_cast<runtime::he::he_seal::HESealCKKSBackend*>(backend.get());

  size_t port = he_seal_ckks_backend->get_port();
  NGRAPH_INFO << "Port " << port;

  boost::asio::io_context io_context;
  tcp::resolver resolver(io_context);
  auto client_endpoints = resolver.resolve("localhost", std::to_string(port));
  tcp::endpoint server_endpoints(tcp::v4(), port);

  auto client = runtime::he::TCPClient(io_context, client_endpoints);

  seal::PublicKey public_key = *he_seal_ckks_backend->get_public_key();

  stringstream stream;
  public_key.save(stream);

  const std::string& pk_string = stream.str();
  const char* pk_data = pk_string.c_str();

  size_t pk_size = sizeof(pk_string);

  NGRAPH_INFO << "pk_size " << pk_size;

  auto message =
      runtime::he::TCPMessage(runtime::he::Datatype::PUBLIC_KEY, 1, pk_size,
                              runtime::he::MPCFunction::NONE, pk_data);

  client.write_message(message);

  io_context.run();
}
