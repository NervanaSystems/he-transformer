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

#pragma once

#include <assert.h>
#include <algorithm>
#include <boost/asio.hpp>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>

#include "ngraph/check.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"
#include "protos/message.pb.h"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {
class TCPMessage {
 public:
  enum { header_length = sizeof(size_t) };
  using data_buffer = std::vector<char>;

  TCPMessage() = default;

  TCPMessage(he_proto::TCPMessage& proto_message)
      : m_proto_message(std::make_shared<he_proto::TCPMessage>(proto_message)) {
  }

  TCPMessage(std::shared_ptr<he_proto::TCPMessage> proto_message)
      : m_proto_message(proto_message) {}

  std::shared_ptr<he_proto::TCPMessage> proto_message() {
    return m_proto_message;
  }
  std::shared_ptr<he_proto::TCPMessage> proto_message() const {
    return m_proto_message;
  }

  static void encode_header(data_buffer& buffer, size_t size) {
    NGRAPH_CHECK(buffer.size() >= header_length, "Buffer too small");
    NGRAPH_INFO << "Encoding header " << size;

    std::memcpy(&buffer[0], &size, header_length);
  }

  static size_t decode_header(const data_buffer& buffer) {
    if (buffer.size() < header_length) {
      return 0;
    }
    size_t body_length = 0;
    std::memcpy(&body_length, &buffer[0], header_length);

    NGRAPH_INFO << "Decoded header body length " << body_length;
    return body_length;
  }

  bool pack(data_buffer& buffer) {
    NGRAPH_CHECK(m_proto_message != nullptr, "Can't pack empy proto message");

    size_t msg_size = m_proto_message->ByteSize();
    NGRAPH_INFO << "Packing buffer with msg_size " << msg_size;

    buffer.resize(header_length + msg_size);
    encode_header(buffer, msg_size);
    return m_proto_message->SerializeToArray(&buffer[header_length], msg_size);
  }

  // buffer => storing proto message
  bool unpack(const data_buffer& buffer) {
    if (!m_proto_message) {
      m_proto_message = std::make_shared<he_proto::TCPMessage>();
    }

    NGRAPH_INFO << "Unpacking from buffer sized " << buffer.size();

    NGRAPH_CHECK(m_proto_message != nullptr, "Can't unpack empty proot");

    bool status = m_proto_message->ParseFromArray(
        &buffer[header_length], buffer.size() - header_length);
    NGRAPH_INFO << "Unpacked";
    return status;
  }

 private:
  std::shared_ptr<he_proto::TCPMessage> m_proto_message;
};
}  // namespace he
}  // namespace ngraph
