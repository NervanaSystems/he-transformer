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

#include <memory>

#include "protos/message.pb.h"

namespace ngraph::runtime::he {
/// \brief Represents a message. A wrapper around pb::TCPMessage
class TCPMessage {
 public:
  enum { header_length = sizeof(size_t) };
  using data_buffer = std::vector<char>;

  /// \brief Creates empty message
  TCPMessage();

  TCPMessage(pb::TCPMessage& proto_message) = delete;

  /// \brief Creates message from given protobuf message
  /// \param[in,out] proto_message Protobuf message to populate TCPMessage
  explicit TCPMessage(pb::TCPMessage&& proto_message);

  /// \brief Returns pointer to udnerlying protobuf message
  std::shared_ptr<pb::TCPMessage> proto_message() const;

  /// \brief Stores a size in the buffer header
  /// \param[in,out] buffer Buffer to write size to
  /// \param[in] size Size to write into buffer
  static void encode_header(data_buffer& buffer, size_t size);

  /// \brief Given a buffer storing a message with the length in the first
  /// header_length bytes, returns the size of the stored buffer
  /// \param[in] buffer Buffer storing a message
  /// \returns size of message stored in buffer
  static size_t decode_header(const data_buffer& buffer);

  /// \brief Writes the message to a buffer
  /// \param[in,out] buffer Buffer to write the message to
  /// \throws ngraph_error if message is empty
  /// \returns Whether or not the operation was successful
  bool pack(data_buffer& buffer);

  /// \brief Writes a given buffer to the message
  /// \param[in] buffer Buffer to read the message from
  /// \returns Whether or not the operation was successful
  bool unpack(const data_buffer& buffer);

 private:
  std::shared_ptr<pb::TCPMessage> m_proto_message;
};
}  // namespace ngraph::runtime::he
