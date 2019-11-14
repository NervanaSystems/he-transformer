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

#include "tcp/tcp_message.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "ngraph/check.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"
#include "protos/message.pb.h"

namespace ngraph::runtime::he {

TCPMessage::TCPMessage() = default;

TCPMessage::TCPMessage(pb::TCPMessage&& proto_message)
    : m_proto_message(
          std::make_shared<pb::TCPMessage>(std::move(proto_message))) {}

std::shared_ptr<pb::TCPMessage> TCPMessage::proto_message() const {
  return m_proto_message;
}

void TCPMessage::encode_header(TCPMessage::data_buffer& buffer, size_t size) {
  NGRAPH_CHECK(buffer.size() >= TCPMessage::header_length, "Buffer too small");
  std::memcpy(&buffer[0], &size, TCPMessage::header_length);
}

size_t TCPMessage::decode_header(const TCPMessage::data_buffer& buffer) {
  if (buffer.size() < TCPMessage::header_length) {
    return 0;
  }
  size_t body_length = 0;
  std::memcpy(&body_length, &buffer[0], TCPMessage::header_length);
  return body_length;
}

bool TCPMessage::pack(TCPMessage::data_buffer& buffer) {
  NGRAPH_CHECK(m_proto_message != nullptr, "Can't pack empty proto message");
  size_t msg_size = m_proto_message->ByteSize();
  buffer.resize(TCPMessage::header_length + msg_size);
  encode_header(buffer, msg_size);
  return m_proto_message->SerializeToArray(&buffer[TCPMessage::header_length],
                                           msg_size);
}

bool TCPMessage::unpack(const TCPMessage::data_buffer& buffer) {
  if (!m_proto_message) {
    m_proto_message = std::make_shared<pb::TCPMessage>();
  }
  return m_proto_message->ParseFromArray(
      &buffer[TCPMessage::header_length],
      buffer.size() - TCPMessage::header_length);
}

}  // namespace ngraph::runtime::he
