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

namespace ngraph::he {

TEST(protobuf, serialize_cipher) {
  pb::TCPMessage message;

  pb::Function f;
  f.set_function("123");
  *message.mutable_function() = f;

  std::stringstream s;
  message.SerializeToOstream(&s);

  pb::TCPMessage deserialize;
  deserialize.ParseFromIstream(&s);

  EXPECT_TRUE(
      google::protobuf::util::MessageDifferencer::Equals(deserialize, message));
}

}  // namespace ngraph::he
