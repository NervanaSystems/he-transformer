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

  auto server_callback = [](const runtime::he::TCPMessage& message) {
    std::cout << "Server callback for message" << std::endl;
    return message;
  };

  auto server =
      runtime::he::TCPServer(io_context, server_endpoints, server_callback);

  auto client_callback = [](const runtime::he::TCPMessage& message) {
    std::cout << "Client callback for message" << std::endl;
    return message;
  };

  auto client =
      runtime::he::TCPClient(io_context, client_endpoints, client_callback);

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

  auto client_callback = [](const runtime::he::TCPMessage& message) {
    std::cout << "Client callback for message" << std::endl;
    return message;
  };

  auto client =
      runtime::he::TCPClient(io_context, client_endpoints, client_callback);

  io_context.run();
}

NGRAPH_TEST(${BACKEND_NAME}, tcp_client_server_init3) {
  auto server_fun = []() {
    try {
      auto backend = runtime::Backend::create("${BACKEND_NAME}");
      auto he_seal_ckks_backend =
          static_cast<runtime::he::he_seal::HESealCKKSBackend*>(backend.get());
    } catch (int e) {
      std::cout << "Exception in server" << std::endl;
    }
  };

  auto client_fun = []() {
    sleep(2);  // Let server start
    size_t port = 34000;
    NGRAPH_INFO << "Client starting " << port;

    boost::asio::io_context io_context;
    tcp::resolver resolver(io_context);
    auto client_endpoints = resolver.resolve("localhost", std::to_string(port));

    auto client_callback = [](const runtime::he::TCPMessage& message) {
      std::cout << "Client callback for message" << std::endl;
      return message;
    };

    auto client =
        runtime::he::TCPClient(io_context, client_endpoints, client_callback);

    sleep(2);  // Let connection happen

    std::thread t([&io_context]() { io_context.run(); });

    NGRAPH_INFO << "Writing message";

    size_t N = 100;
    void* x = malloc(N);
    memset(x, 0, N);
    assert(x != nullptr);
    auto message = runtime::he::TCPMessage(
        runtime::he::MessageType::public_key_request, 0, N, (char*)x);
    client.write_message(message);

    sleep(5);  // Let message be handled
    client.close();

    t.join();
    // io_context.run();

    // sleep(1);  // Let message arrive
  };

  /*auto pid = fork();
  if (pid == 0) {
    sleep(1);
    NGRAPH_INFO << "Client fun";
    client_fun();
    sleep(1);
  } else {
    NGRAPH_INFO << "SErver fun";
    server_fun();
  } */

  std::thread t1(client_fun);
  std::thread t2(server_fun);

  sleep(1);

  t1.join();
  t2.join();
}
