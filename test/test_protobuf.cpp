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
#include "tcp/tcp_message.hpp"
#include "message.pb.h"
#include "seal/seal.h"

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

TEST(new_tcp_message, from_parms) {
  seal::EncryptionParameters parms(seal::scheme_type::CKKS);
  size_t poly_modulus_degree = 8192;
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(
      seal::CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));

  std::stringstream param_stream;
  seal::EncryptionParameters::Save(parms,param_stream);

  he_proto::EncryptionParameters proto_parms;
  *proto_parms.mutable_encryption_parameters() = param_stream.str();

  he_proto::TCPMessage proto_msg;
  *proto_msg.mutable_encryption_parameters() = proto_parms;

  ngraph::he::NewTCPMessage tcp_message(proto_msg);
}