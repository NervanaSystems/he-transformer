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

#include <google/protobuf/util/message_differencer.h>

#include <chrono>
#include <memory>

#include "gtest/gtest.h"
#include "protos/message.pb.h"
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "tcp/tcp_message.hpp"
#include "util/test_tools.hpp"

TEST(protobuf, trivial) { EXPECT_EQ(1, 1); }

TEST(protobuf, serialize_cipher) {
  ngraph::he::pb::TCPMessage message;

  ngraph::he::pb::Function f;
  f.set_function("123");
  *message.mutable_function() = f;

  std::stringstream s;
  message.SerializeToOstream(&s);

  ngraph::he::pb::TCPMessage deserialize;
  deserialize.ParseFromIstream(&s);

  EXPECT_TRUE(
      google::protobuf::util::MessageDifferencer::Equals(deserialize, message));
}

TEST(tcp_message, create) {
  ngraph::he::pb::TCPMessage proto_msg;
  ngraph::he::pb::Function f;
  f.set_function("123");
  *proto_msg.mutable_function() = f;
  std::stringstream s;
  proto_msg.SerializeToOstream(&s);
  ngraph::he::pb::TCPMessage tcp_message(std::move(proto_msg));
  EXPECT_EQ(1, 1);
}

TEST(tcp_message, encode_decode) {
  using data_buffer = std::vector<char>;
  data_buffer buffer;
  buffer.resize(20);

  size_t encode_size = 10;
  ngraph::he::TCPMessage::encode_header(buffer, encode_size);
  size_t decoded_size = ngraph::he::TCPMessage::decode_header(buffer);
  EXPECT_EQ(decoded_size, encode_size);
}

TEST(tcp_message, pack_unpack) {
  using data_buffer = std::vector<char>;

  ngraph::he::pb::TCPMessage proto_msg;
  ngraph::he::pb::Function f;
  f.set_function("123");
  *proto_msg.mutable_function() = f;
  std::stringstream s;
  proto_msg.SerializeToOstream(&s);
  ngraph::he::TCPMessage message1(std::move(proto_msg));

  data_buffer buffer;
  message1.pack(buffer);

  ngraph::he::TCPMessage message2;
  message2.unpack(buffer);

  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      *message1.proto_message(), *message2.proto_message()));
}
