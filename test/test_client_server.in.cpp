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

NGRAPH_TEST(${BACKEND_NAME}, tcp_message_encode_request) {
  auto message =
      runtime::he::TCPMessage(runtime::he::MessageType::public_key_request);

  runtime::he::TCPMessage message2 = message;
  message2.decode_header();

  EXPECT_EQ(message.message_type(), message2.message_type());
  EXPECT_EQ(message.count(), message2.count());
  EXPECT_EQ(message.num_bytes(), message2.num_bytes());
  EXPECT_EQ(message.data_size(), message2.data_size());
}

NGRAPH_TEST(${BACKEND_NAME}, tcp_message_encode) {
  size_t count = 3;
  size_t element_size = 10;
  size_t size = count * element_size;
  void* data = malloc(size);
  std::memset(data, 7, size);  // Set data to have value 7
  assert(data != nullptr);

  auto message = runtime::he::TCPMessage(runtime::he::MessageType::none, count,
                                         size, (char*)data);

  runtime::he::TCPMessage message2 = message;
  message2.decode_header();
  message2.decode_body();

  EXPECT_EQ(std::memcmp(message.data_ptr(), data, size), 0);

  EXPECT_EQ(message.message_type(), message2.message_type());
  EXPECT_EQ(message.count(), message2.count());
  EXPECT_EQ(message.num_bytes(), message2.num_bytes());
  EXPECT_EQ(message.data_size(), message2.data_size());

  message = runtime::he::TCPMessage(runtime::he::MessageType::public_key, count,
                                    size, (char*)data);
  message2 = message;
  message2.decode_header();
  message2.decode_body();

  EXPECT_EQ(message.message_type(), message2.message_type());
  EXPECT_EQ(message.count(), message2.count());
  EXPECT_EQ(message.num_bytes(), message2.num_bytes());
  EXPECT_EQ(message.data_size(), message2.data_size());
}

NGRAPH_TEST(${BACKEND_NAME}, tcp_client_server_init) {
  Shape shape{1};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b =
      make_shared<op::Constant>(element::f32, shape, std::vector<float>{2.3});
  auto t = make_shared<op::Add>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{a});

  auto server_fun = [&f]() {
    try {
      NGRAPH_INFO << "Server starting";
      auto backend = runtime::Backend::create("${BACKEND_NAME}");
      auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

      auto handle = backend->compile(f);

      he_backend->start_server();

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

    // client.request_public_key(); <-- handler processes response
    // client.request_context();    <-- handler processes response
    // encrypt_data();
    // client.write_message("Perform inference"); <-- handler processes response
    //                                                until Result
    // client.print_results()
    // client.close_connection()

    sleep(5);  // Let message be handled
    client.close_connection();
  };
  std::thread t1(client_fun);
  std::thread t2(server_fun);

  sleep(5);

  t1.join();
  t2.join();
}
