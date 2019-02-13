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
#include "seal/he_seal_client.hpp"
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

NGRAPH_TEST(${BACKEND_NAME}, tcp_client_server_init3) {
  auto server_fun = []() {
    try {
      NGRAPH_INFO << "Server starting";
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
    auto client = runtime::he::HESealClient(io_context, client_endpoints);

    // sleep(2);  // Let connection happen

    /*NGRAPH_INFO << "Writing message";

    size_t N = 100;
    void* x = malloc(N);
    memset(x, 0, N);
    assert(x != nullptr);
    auto message = runtime::he::TCPMessage(
        runtime::he::MessageType::public_key_request, 0, N, (char*)x);
    client.write_message(message); */

    sleep(5);  // Let message be handled
    client.close_connection();
  };
  std::thread t1(client_fun);
  std::thread t2(server_fun);

  sleep(5);

  t1.join();
  t2.join();
}
