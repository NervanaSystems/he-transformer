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
#include <memory>

#include "gtest/gtest.h"
#include "helloworld.pb.h"
#include "message.pb.h"
#include "seal/seal.h"
#include "tcp/tcp_message.hpp"

using namespace std;

TEST(protobuf, trivial) { EXPECT_EQ(1, 1); }

TEST(protobuf, serialize) {
  helloworld::HelloRequest request;
  request.set_name("name");
  EXPECT_EQ(request.name(), "name");

  std::stringstream s;
  request.SerializeToOstream(&s);

  helloworld::HelloRequest deserialize;
  deserialize.ParseFromIstream(&s);

  EXPECT_EQ(deserialize.name(), request.name());
}

TEST(protobuf, serialize_cipher) {
  he_proto::TCPMessage message;

  he_proto::Function f;
  f.set_function("123");
  *message.mutable_function() = f;

  std::stringstream s;
  message.SerializeToOstream(&s);

  he_proto::TCPMessage deserialize;
  deserialize.ParseFromIstream(&s);

  EXPECT_TRUE(
      google::protobuf::util::MessageDifferencer::Equals(deserialize, message));
}

TEST(new_tcp_message, create) {
  he_proto::TCPMessage proto_msg;
  he_proto::Function f;
  f.set_function("123");
  *proto_msg.mutable_function() = f;
  std::stringstream s;
  proto_msg.SerializeToOstream(&s);

  ngraph::he::NewTCPMessage tcp_message(proto_msg);
}

TEST(new_tcp_message, encode_decode) {
  using data_buffer = std::vector<char>;

  data_buffer buffer;
  buffer.resize(20);

  size_t encode_size = 10;
  ngraph::he::NewTCPMessage::encode_header(buffer, encode_size);
  size_t decoded_size = ngraph::he::NewTCPMessage::decode_header(buffer);

  EXPECT_EQ(decoded_size, encode_size);
}
