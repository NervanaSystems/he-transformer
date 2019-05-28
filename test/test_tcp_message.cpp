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
#include "ngraph/ngraph.hpp"
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

NGRAPH_TEST(tcp_message, encode_request) {
  auto m = ngraph::he::TCPMessage(ngraph::he::MessageType::result_request);

  ngraph::he::TCPMessage m2;
  // read header
  std::memcpy(m2.header_ptr(), m.header_ptr(),
              ngraph::he::TCPMessage::header_length);
  m2.decode_header();
  // read body
  std::memcpy(m2.body_ptr(), m.body_ptr(), m2.body_length());
  m2.decode_body();

  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(),
                        ngraph::he::TCPMessage::header_length),
            0);
  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(), m2.num_bytes()), 0);
  EXPECT_EQ(m.message_type(), m2.message_type());
  EXPECT_EQ(m.count(), m2.count());
  EXPECT_EQ(m.num_bytes(), m2.num_bytes());
  EXPECT_EQ(m.data_size(), m2.data_size());
}

NGRAPH_TEST(tcp_message, copy) {
  size_t count = 3;
  size_t element_size = 10;
  size_t size = count * element_size;
  void* data = malloc(size);
  std::memset(data, 7, size);  // Set data to have value 7
  assert(data != nullptr);

  auto m = ngraph::he::TCPMessage(ngraph::he::MessageType::eval_key, count,
                                   size, (char*)data);
  ngraph::he::TCPMessage m2{m};

  EXPECT_EQ(m.message_type(), m2.message_type());
  EXPECT_EQ(m.count(), m2.count());
  EXPECT_EQ(m.num_bytes(), m2.num_bytes());
  EXPECT_EQ(m.data_size(), m2.data_size());

  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(),
                        ngraph::he::TCPMessage::header_length),
            0);
  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(), m2.num_bytes()), 0);
}

NGRAPH_TEST(tcp_message, encode) {
  size_t count = 3;
  size_t element_size = 10;
  size_t size = count * element_size;
  void* data = malloc(size);
  std::memset(data, 7, size);  // Set data to have value 7
  assert(data != nullptr);

  auto m = ngraph::he::TCPMessage(ngraph::he::MessageType::none, count, size,
                                   (char*)data);

  ngraph::he::TCPMessage m2;
  // read header
  std::memcpy(m2.header_ptr(), m.header_ptr(),
              ngraph::he::TCPMessage::header_length);
  m2.decode_header();
  // read body
  std::memcpy(m2.body_ptr(), m.body_ptr(), m2.body_length());
  m2.decode_body();

  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(),
                        ngraph::he::TCPMessage::header_length),
            0);
  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(), m2.num_bytes()), 0);
  EXPECT_EQ(m.message_type(), m2.message_type());
  EXPECT_EQ(m.count(), m2.count());
  EXPECT_EQ(m.num_bytes(), m2.num_bytes());
  EXPECT_EQ(m.data_size(), m2.data_size());
}

NGRAPH_TEST(tcp_message, encode_large) {
  size_t count = 784;
  size_t element_size = 262217;
  size_t size = count * element_size;
  void* data = malloc(size);
  std::memset(data, 7, size);  // Set data to have value 7
  assert(data != nullptr);

  auto m = ngraph::he::TCPMessage(ngraph::he::MessageType::none, count, size,
                                   (char*)data);

  ngraph::he::TCPMessage m2;
  std::memcpy(m2.header_ptr(), m.header_ptr(),
              ngraph::he::TCPMessage::header_length);
  m2.decode_header();
  std::memcpy(m2.body_ptr(), m.body_ptr(), m2.body_length());
  m2.decode_body();

  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(),
                        ngraph::he::TCPMessage::header_length),
            0);
  EXPECT_EQ(std::memcmp(m.header_ptr(), m2.header_ptr(), m2.num_bytes()), 0);
  EXPECT_EQ(m.message_type(), m2.message_type());
  EXPECT_EQ(m.count(), m2.count());
  EXPECT_EQ(m.num_bytes(), m2.num_bytes());
  EXPECT_EQ(m.data_size(), m2.data_size());

  free(data);
}