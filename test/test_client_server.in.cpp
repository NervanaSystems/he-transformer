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
#include <thread>

#include "boost/asio.hpp"
#include "he_backend.hpp"
#include "ngraph/ngraph.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/he_seal_client.hpp"
#include "tcp/tcp_client.hpp"
#include "tcp/tcp_message.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, tcp_message_encode_request) {
  auto m = runtime::he::TCPMessage(runtime::he::MessageType::result_request);

  runtime::he::TCPMessage m2;
  // read header
  std::memcpy(m2.header_ptr(), m.header_ptr(),
              runtime::he::TCPMessage::header_length);
  m2.decode_header();
  // read body
  std::memcpy(m2.body_ptr(), m.body_ptr(), m2.body_length());
  m2.decode_body();

  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(),
                        runtime::he::TCPMessage::header_length),
            0);
  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(), m2.num_bytes()), 0);
  EXPECT_EQ(m.message_type(), m2.message_type());
  EXPECT_EQ(m.count(), m2.count());
  EXPECT_EQ(m.num_bytes(), m2.num_bytes());
  EXPECT_EQ(m.data_size(), m2.data_size());
}

NGRAPH_TEST(${BACKEND_NAME}, tcp_message_copy) {
  size_t count = 3;
  size_t element_size = 10;
  size_t size = count * element_size;
  void* data = malloc(size);
  std::memset(data, 7, size);  // Set data to have value 7
  assert(data != nullptr);

  auto m = runtime::he::TCPMessage(runtime::he::MessageType::eval_key, count,
                                   size, (char*)data);
  runtime::he::TCPMessage m2{m};

  EXPECT_EQ(m.message_type(), m2.message_type());
  EXPECT_EQ(m.count(), m2.count());
  EXPECT_EQ(m.num_bytes(), m2.num_bytes());
  EXPECT_EQ(m.data_size(), m2.data_size());

  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(),
                        runtime::he::TCPMessage::header_length),
            0);
  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(), m2.num_bytes()), 0);
}

/* NGRAPH_TEST(${BACKEND_NAME}, tcp_message_move) {
  size_t count = 3;
  size_t element_size = 10;
  size_t size = count * element_size;
  void* data = malloc(size);
  std::memset(data, 7, size);  // Set data to have value 7
  assert(data != nullptr);

  // m constructed via move constructor
  runtime::he::TCPMessage m{std::move(runtime::he::TCPMessage(
      runtime::he::MessageType::eval_key, count, size, (char*)data))};
  runtime::he::TCPMessage m2 = runtime::he::TCPMessage(
      runtime::he::MessageType::eval_key, count, size, (char*)data);

  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(),
                        runtime::he::TCPMessage::header_length),
            0);
  EXPECT_EQ(m.num_bytes(), m2.num_bytes());
  EXPECT_EQ(m.message_type(), m2.message_type());
  EXPECT_EQ(m.count(), m2.count());
  EXPECT_EQ(m.data_size(), m2.data_size());
  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(), m2.num_bytes()), 0);

  NGRAPH_INFO << "Freeing data";

  free(data);
} */

NGRAPH_TEST(${BACKEND_NAME}, tcp_message_encode) {
  size_t count = 3;
  size_t element_size = 10;
  size_t size = count * element_size;
  void* data = malloc(size);
  std::memset(data, 7, size);  // Set data to have value 7
  assert(data != nullptr);

  auto m = runtime::he::TCPMessage(runtime::he::MessageType::none, count, size,
                                   (char*)data);

  runtime::he::TCPMessage m2;
  // read header
  std::memcpy(m2.header_ptr(), m.header_ptr(),
              runtime::he::TCPMessage::header_length);
  m2.decode_header();
  // read body
  std::memcpy(m2.body_ptr(), m.body_ptr(), m2.body_length());
  m2.decode_body();

  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(),
                        runtime::he::TCPMessage::header_length),
            0);
  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(), m2.num_bytes()), 0);
  EXPECT_EQ(m.message_type(), m2.message_type());
  EXPECT_EQ(m.count(), m2.count());
  EXPECT_EQ(m.num_bytes(), m2.num_bytes());
  EXPECT_EQ(m.data_size(), m2.data_size());
}

NGRAPH_TEST(${BACKEND_NAME}, tcp_message_encode_large) {
  size_t count = 784;
  size_t element_size = 262217;
  size_t size = count * element_size;
  void* data = malloc(size);
  std::memset(data, 7, size);  // Set data to have value 7
  assert(data != nullptr);

  NGRAPH_INFO << "Size " << size;

  auto m = runtime::he::TCPMessage(runtime::he::MessageType::none, count, size,
                                   (char*)data);
  NGRAPH_INFO << "Made message";

  runtime::he::TCPMessage m2;
  // read header
  std::memcpy(m2.header_ptr(), m.header_ptr(),
              runtime::he::TCPMessage::header_length);
  m2.decode_header();
  // read body
  std::memcpy(m2.body_ptr(), m.body_ptr(), m2.body_length());
  m2.decode_body();

  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(),
                        runtime::he::TCPMessage::header_length),
            0);
  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(), m2.num_bytes()), 0);
  EXPECT_EQ(m.message_type(), m2.message_type());
  EXPECT_EQ(m.count(), m2.count());
  EXPECT_EQ(m.num_bytes(), m2.num_bytes());
  EXPECT_EQ(m.data_size(), m2.data_size());
}

NGRAPH_TEST(${BACKEND_NAME}, tcp_init) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, tcp_init2) {
  auto server_fun = []() {
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
  };

  std::thread t2(server_fun);
  t2.join();
}

NGRAPH_TEST(${BACKEND_NAME}, tcp_ng_tf) {
  Shape shape{2, 3};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Constant>(
      element::f32, shape, std::vector<float>{1.1, 1.2, 1.3, 1.4, 1.5, 1.6});
  auto t = make_shared<op::Add>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{a});

  auto server_fun = [&f, &shape]() {
    try {
      NGRAPH_INFO << "Creating backend";
      auto backend = runtime::Backend::create("${BACKEND_NAME}");
      NGRAPH_INFO << "Compiling function";
      auto handle = backend->compile(f);
      sleep(5);
      NGRAPH_INFO << "Destructing server";

    } catch (std::system_error& e) {
      NGRAPH_INFO << "Exception in server";
    }
  };

  std::vector<float> results;
  auto client_fun = [&results]() {
    try {
      sleep(1);  // Let server start
      size_t port = 34000;
      std::vector<float> inputs{1, 2, 3, 4, 5, 6};
      boost::asio::io_context io_context;
      tcp::resolver resolver(io_context);
      auto client_endpoints =
          resolver.resolve("localhost", std::to_string(port));
      NGRAPH_INFO << "Creating client";
      auto client =
          runtime::he::HESealClient(io_context, client_endpoints, inputs);

      NGRAPH_INFO << "Waiting for client results";

      sleep(1);
      /*while (!client.is_done()) {
        sleep(1);
      }
      NGRAPH_INFO << "Getting client results";
      results = client.get_results();*/

      NGRAPH_INFO << "Closing client";

    } catch (std::system_error& e) {
      NGRAPH_INFO << "Exception in client";
    }
  };
  std::thread t1(client_fun);
  std::thread t2(server_fun);
  t1.join();
  NGRAPH_INFO << "Client function ended";
  t2.join();
  NGRAPH_INFO << "Server function ended";

  EXPECT_TRUE(
      all_close(results, std::vector<float>{2.1, 3.2, 4.3, 5.4, 6.5, 7.6}));
}