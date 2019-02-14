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
#include <cstring>
#include <iostream>
#include <memory>
#include <set>

namespace ngraph {
namespace runtime {
namespace he {
enum class MessageType {
  none,
  public_key_ack,
  public_key,
  public_key_request,
  execute,
  parameter_shape_request,
  parameter_shape,
  relu_request,
  relu,
  result
};

inline std::string message_type_to_string(const MessageType& type) {
  switch (type) {
    case MessageType::none:
      return "none";
      break;
    case MessageType::public_key_ack:
      return "public_key_ack";
      break;
    case MessageType::public_key:
      return "public_key";
      break;
    case MessageType::public_key_request:
      return "public_key_request";
      break;
    case MessageType::execute:
      return "execute";
      break;
    case MessageType::parameter_shape:
      return "parameter_shape";
      break;
    case MessageType::parameter_shape_request:
      return "parameter_shape_request";
      break;
    case MessageType::result:
      return "result";
      break;
    default:
      return "Unknown message type";
  }
}

class TCPMessage {
 public:
  enum { header_length = sizeof(size_t) };
  enum { max_body_length = 400000 };

  TCPMessage(const MessageType type) : m_type(type) {
    std::set<MessageType> request_types{
        MessageType::public_key_request, MessageType::relu_request,
        MessageType::public_key_ack, MessageType::parameter_shape_request};

    if (request_types.find(type) == request_types.end()) {
      throw std::invalid_argument("Request type not valid");
    }
    m_bytes = header_length + sizeof(MessageType);
    m_body_length = sizeof(MessageType);
    m_count = 0;
    m_data_size = 0;
    m_element_size = 0;

    encode_header();
    encode_message_type();
  }

  TCPMessage()
      : m_bytes(header_length),
        m_body_length(0),
        m_type(MessageType::none),
        m_element_size(0),
        m_count(0) {}

  // @brief Describes TCP messages of the form:
  // header        | message_type | count        | data
  // ^- header_ptr   ^- body_ptr    ^- count_ptr   ^- data_ptr
  // @param count number of elements of data
  // @param size number of bytes of data in message. Must be a multiple of
  // count
  TCPMessage(const MessageType type, const size_t count, const size_t size,
             const char* data)
      : m_type(type), m_count(count), m_data_size(size) {
    if (count != 0 && size % count != 0) {
      std::cout << "size " << size << " count " << count << std::endl;
      throw std::invalid_argument("Size must be a multiple of count");
    }
    std::cout << "Creating message with data size " << size << std::endl;

    m_bytes = header_length;
    m_bytes += sizeof(MessageType);
    m_bytes += sizeof(count);
    m_bytes += size;

    m_body_length = m_bytes - header_length;

    if (m_body_length > max_body_length) {
      throw std::invalid_argument("Size " + std::to_string(m_body_length) +
                                  " too large");
    }

    m_element_size = size / count;

    encode_header();
    encode_message_type();
    encode_count_and_data(data);
  }

  size_t count() { return m_count; }

  const size_t count() const { return m_count; }

  size_t element_size() { return m_element_size; }

  const size_t element_size() const { return m_element_size; }

  size_t num_bytes() { return m_bytes; }

  const size_t num_bytes() const { return m_bytes; }

  size_t data_size() { return m_data_size; }

  const size_t data_size() const { return m_data_size; }

  size_t body_length() const { return m_body_length; }

  MessageType message_type() { return m_type; }

  const MessageType message_type() const { return m_type; }

  char* header_ptr() { return m_data; }

  const char* header_ptr() const { return m_data; }

  char* body_ptr() { return m_data + header_length; }

  const char* body_ptr() const { return m_data + header_length; }

  char* count_ptr() { return m_data + header_length + sizeof(MessageType); }

  const char* count_ptr() const {
    return m_data + header_length + sizeof(MessageType);
  }

  char* data_ptr() {
    return m_data + header_length + sizeof(MessageType) + sizeof(size_t);
  }

  const char* data_ptr() const {
    return m_data + header_length + sizeof(MessageType) + sizeof(size_t);
  }

  void encode_header() {
    assert(header_length == sizeof(m_bytes));
    // std::cout << "Encoding header " << m_body_length << std::endl;

    char header[header_length + 1] = "";
    std::sprintf(header, "%4d", static_cast<int>(m_body_length));
    std::memcpy(m_data, header, header_length);
  }

  void encode_message_type() {
    // std::cout << "Encoding message type "
    //          << message_type_to_string(m_type).c_str() << std::endl;

    std::memcpy(body_ptr(), &m_type, sizeof(MessageType));
  }

  void encode_count_and_data(const char* data) {
    // Copy count
    std::memcpy(count_ptr(), &m_count, sizeof(size_t));

    // Copy data
    std::memcpy(data_ptr(), data, m_data_size);
  }

  // Given m_data, parses to find m_datatype, m_count
  bool decode_body() {
    MessageType type;
    // Decode message type
    std::memcpy(&type, body_ptr(), sizeof(MessageType));
    // std::cout << "Decoded message type " <<
    // message_type_to_string(type).c_str()
    //          << std::endl;
    m_type = type;

    if (m_body_length > sizeof(MessageType)) {
      // Decode message count
      std::memcpy(&m_count, count_ptr(), sizeof(size_t));
      // Decode data size
      m_data_size = m_body_length - sizeof(MessageType) - sizeof(size_t);

      m_element_size = m_data_size / m_count;

      // std::cout << "Decoded message count " << m_count << std::endl;
      // std::cout << "Decoded m_data_size " << m_data_size << std::endl;
    } else {
      m_data_size = 0;
      m_element_size = 0;
    }
  }

  bool decode_header() {
    char header[header_length + 1] = "";
    std::strncat(header, m_data, header_length);
    m_body_length = std::atoi(header);
    if (m_body_length > max_body_length) {
      m_body_length = 0;
      std::cout << "Body length " << m_body_length << " too large" << std::endl;
      return false;
    }
    // std::cout << "Decoding header; message size " << m_body_length <<
    // std::endl;
    return true;
  }

 private:
  size_t m_bytes;         // How many bytes in the message
  size_t m_body_length;   // How many bytes in message body
  size_t m_count;         // Number of datatype in message
  size_t m_data_size;     // Nubmer of bytes in data part of message
  size_t m_element_size;  // How many bytes per datatype in message

  MessageType m_type;  // What data is being transmitted
  char m_data[header_length + max_body_length];

  // void* m_data_ptr;
};  // namespace he
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
