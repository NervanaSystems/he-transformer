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
  auto m =
      runtime::he::TCPMessage(runtime::he::MessageType::public_key_request);

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

  auto m = runtime::he::TCPMessage(runtime::he::MessageType::public_key, count,
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

NGRAPH_TEST(${BACKEND_NAME}, tcp_message_move) {
  size_t count = 3;
  size_t element_size = 10;
  size_t size = count * element_size;
  void* data = malloc(size);
  std::memset(data, 7, size);  // Set data to have value 7
  assert(data != nullptr);

  // m constructed via move constructor
  runtime::he::TCPMessage m{std::move(runtime::he::TCPMessage(
      runtime::he::MessageType::public_key, count, size, (char*)data))};
  runtime::he::TCPMessage m2 = runtime::he::TCPMessage(
      runtime::he::MessageType::public_key, count, size, (char*)data);

  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(),
                        runtime::he::TCPMessage::header_length),
            0);
  EXPECT_EQ(m.num_bytes(), m2.num_bytes());
  EXPECT_EQ(m.message_type(), m2.message_type());
  EXPECT_EQ(m.count(), m2.count());
  EXPECT_EQ(m.data_size(), m2.data_size());
  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(), m2.num_bytes()), 0);
}

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
/*
NGRAPH_TEST(${BACKEND_NAME}, tcp_client_server_init) {
  Shape shape{2, 3};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Constant>(
      element::f32, shape, std::vector<float>{1.1, 1.2, 1.3, 1.4, 1.5, 1.6});
  auto t = make_shared<op::Add>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{a});

  auto server_fun = [&f]() {
    try {
      NGRAPH_INFO << "Creating backend";
      auto backend = runtime::Backend::create("${BACKEND_NAME}");
      auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());
      auto handle = backend->compile(f);
      NGRAPH_INFO << "Starting server";
      he_backend->start_server();
    } catch (std::system_error& e) {
      NGRAPH_INFO<< "Exception in server";
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
      auto client =
          runtime::he::HESealClient(io_context, client_endpoints, inputs);
      while (!client.is_done()) {
        sleep(1);
      }
      results = client.get_results();
    } catch (std::system_error& e) {
      NGRAPH_INFO<< "Exception in client";
    }
  };
  std::thread t1(client_fun);
  std::thread t2(server_fun);
  t1.join();
  t2.join();

  EXPECT_TRUE(
      all_close(results, std::vector<float>{2.1, 3.2, 4.3, 5.4, 6.5, 7.6}));
  std::cout;
}

NGRAPH_TEST(${BACKEND_NAME}, tcp_client_server_relu) {
  int N = 100;
  std::vector<float> inputs;
  for (int i = 0; i < N; ++i) {
    if (i % 2 == 0) {
      inputs.emplace_back(i);
    } else {
      inputs.emplace_back(-i);
    }
  }

  std::vector<float> expected_results;
  std::transform(inputs.begin(), inputs.end(),
                 std::back_inserter(expected_results), [](float x) -> float {
                   if (x > 0) {
                     return x;
                   } else {
                     return 0.f;
                   }
                 });
  Shape shape{inputs.size()};

  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Relu>(a);
  auto f = make_shared<Function>(t, ParameterVector{a});

  auto server_fun = [&f]() {
    try {
      NGRAPH_INFO << "Creating backend";
      auto backend = runtime::Backend::create("${BACKEND_NAME}");
      auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());
      auto handle = backend->compile(f);
      NGRAPH_INFO << "Starting server";
      he_backend->start_server();
    } catch (std::system_error& e) {
      NGRAPH_INFO<< "Exception in server";
    }
  };

  std::vector<float> results;
  auto client_fun = [&inputs, &results]() {
    try {
      // Wait for server to start
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      size_t port = 34000;

      boost::asio::io_context io_context;
      tcp::resolver resolver(io_context);
      auto client_endpoints =
          resolver.resolve("localhost", std::to_string(port));
      auto client =
          runtime::he::HESealClient(io_context, client_endpoints, inputs);
      while (!client.is_done()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      results = client.get_results();
    } catch (std::system_error& e) {
      NGRAPH_INFO<< "Exception in client";
    }
  };
  std::thread t1(client_fun);
  std::thread t2(server_fun);
  t1.join();
  t2.join();

  EXPECT_TRUE(all_close(results, expected_results));
  std::cout;
}

NGRAPH_TEST(${BACKEND_NAME}, tcp_client_server_relu2) {
  int N = 6;
  std::vector<float> inputs;
  std::vector<float> const_a;
  std::vector<float> const_b;
  std::vector<float> expected_results;
  auto relu = [](float x) -> float { return x > 0 ? x : 0.f; };

  for (int i = 0; i < N; ++i) {
    float x;
    if (i % 2 == 0) {
      x = i;
      inputs.emplace_back(i);
    } else {
      x = -i;
      inputs.emplace_back(-i);
    }
    float a = i + 0.1;
    float b = i + 1;
    const_a.emplace_back(a);
    const_b.emplace_back(b);

    expected_results.emplace_back(relu(relu(x + a) + b));
  }

  Shape shape{inputs.size()};

  NGRAPH_INFO << "Expected results";
  for (const auto& elem : expected_results) {
    NGRAPH_INFO << elem;
  }

  auto x = make_shared<op::Parameter>(element::f32, shape);
  auto a = make_shared<op::Constant>(element::f32, shape, const_a);
  auto b = make_shared<op::Constant>(element::f32, shape, const_b);
  auto relu1 = make_shared<op::Relu>(x + a);
  auto t = make_shared<op::Relu>(relu1 + b);
  auto f = make_shared<Function>(t, ParameterVector{x});

  auto server_fun = [&f]() {
    try {
      NGRAPH_INFO << "Creating backend";
      auto backend = runtime::Backend::create("${BACKEND_NAME}");
      auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());
      auto handle = backend->compile(f);
      NGRAPH_INFO << "Starting server";
      he_backend->start_server();
    } catch (std::system_error& e) {
      NGRAPH_INFO<< "Exception in server";
    }
  };

  std::vector<float> results;
  auto client_fun = [&inputs, &results]() {
    try {
      // Wait for server to start
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      size_t port = 34000;

      boost::asio::io_context io_context;
      tcp::resolver resolver(io_context);
      auto client_endpoints =
          resolver.resolve("localhost", std::to_string(port));
      auto client =
          runtime::he::HESealClient(io_context, client_endpoints, inputs);
      while (!client.is_done()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      results = client.get_results();
    } catch (std::system_error& e) {
      NGRAPH_INFO<< "Exception in client";
    }
  };
  std::thread t1(client_fun);
  std::thread t2(server_fun);
  t1.join();
  t2.join();

  EXPECT_TRUE(all_close(results, expected_results));
  std::cout;
}
*/

NGRAPH_TEST(${BACKEND_NAME}, tcp_init) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, tcp_init2) {
  auto server_fun = [&]() {
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    sleep(1);
  };

  std::thread t2(server_fun);
  sleep(1);
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

      /* auto handle = backend->compile(f);
      NGRAPH_INFO << "Creating tensor";
      auto result = backend->create_tensor(element::f32, shape);
      NGRAPH_INFO << "Created tensor"; */
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