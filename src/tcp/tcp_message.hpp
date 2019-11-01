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

#include <cstring>
#include <memory>
#include <sstream>
#include <string>

#include "ngraph/check.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"
#include "protos/message.pb.h"

namespace ngraph {
namespace he {
/// \brief Represents a message. A wrapper around proto::TCPMessage
class TCPMessage {
 public:
  enum { header_length = sizeof(size_t) };
  using data_buffer = std::vector<char>;

  /// \brief Creates empty message
  TCPMessage() = default;

  /// \brief Creates message from given protobuf message
  /// \param[in] proto_message Protobuf message to populate TCPMessage
  TCPMessage(proto::TCPMessage& proto_message) = delete;

  /// \brief Creates message from given protobuf message
  /// \param[in,out] proto_message Protobuf message to populate TCPMessage
  TCPMessage(proto::TCPMessage&& proto_message)
      : m_proto_message(
            std::make_shared<proto::TCPMessage>(std::move(proto_message))) {}

  /// \brief Creates message from given protobuf message
  /// \param[in,out] proto_message Protobuf message to populate TCPMessage
  TCPMessage(std::shared_ptr<proto::TCPMessage> proto_message)
      : m_proto_message(proto_message) {}

  /// \brief Returns pointer to udnerlying protobuf message
  std::shared_ptr<proto::TCPMessage> proto_message() { return m_proto_message; }

  /// \brief Returns pointer to udnerlying protobuf message
  std::shared_ptr<proto::TCPMessage> proto_message() const {
    return m_proto_message;
  }

  /// \brief Stores a size in the buffer header
  /// \param[in,out] buffer Buffer to write size to
  /// \param[in] size Size to write into buffer
  static void encode_header(data_buffer& buffer, size_t size) {
    NGRAPH_CHECK(buffer.size() >= header_length, "Buffer too small");
    std::memcpy(&buffer[0], &size, header_length);
  }

  /// \brief Given a buffer storing a message with the length in the first
  /// header_length bytes, returns the size of the stored buffer \param[in]
  /// buffer Buffer storing a message \returns size of message stored in buffer
  static size_t decode_header(const data_buffer& buffer) {
    if (buffer.size() < header_length) {
      return 0;
    }
    size_t body_length = 0;
    std::memcpy(&body_length, &buffer[0], header_length);
    return body_length;
  }

  /// \brief Writes the message to a buffer
  /// \param[in,out] buffer Buffer to write the message to
  /// \throws ngraph_error if message is empty
  /// \returns Whether or not the operation was successful
  bool pack(data_buffer& buffer) {
    NGRAPH_CHECK(m_proto_message != nullptr, "Can't pack empty proto message");
    size_t msg_size = m_proto_message->ByteSize();
    buffer.resize(header_length + msg_size);
    encode_header(buffer, msg_size);
    return m_proto_message->SerializeToArray(&buffer[header_length], msg_size);
  }

  /// \brief Writes a given buffer to the message
  /// \param[in] buffer Buffer to read the message from
  /// \returns Whether or not the operation was successful
  bool unpack(const data_buffer& buffer) {
    if (!m_proto_message) {
      m_proto_message = std::make_shared<proto::TCPMessage>();
    }
    return m_proto_message->ParseFromArray(&buffer[header_length],
                                           buffer.size() - header_length);
  }

 private:
  std::shared_ptr<proto::TCPMessage> m_proto_message;
};
}  // namespace he
}  // namespace ngraph
