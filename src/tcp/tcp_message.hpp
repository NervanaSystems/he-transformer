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
#include <cstring>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>

namespace ngraph {
namespace runtime {
namespace he {
enum class MessageType {
  none,
  encryption_parameters,
  public_key_ack,
  public_key,
  public_key_request,
  execute,
  execute_done,
  parameter_shape_request,
  parameter_shape,
  relu_request,
  relu,
  result,
  result_request
};

inline std::string message_type_to_string(const MessageType& type) {
  switch (type) {
    case MessageType::none:
      return "none";
      break;
    case MessageType::encryption_parameters:
      return "encryption_parameters";
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
    case MessageType::execute_done:
      return "execute_done";
      break;
    case MessageType::parameter_shape:
      return "parameter_shape";
      break;
    case MessageType::parameter_shape_request:
      return "parameter_shape_request";
      break;
    case MessageType::relu:
      return "relu";
      break;
    case MessageType::relu_request:
      return "relu_request";
      break;
    case MessageType::result:
      return "result";
      break;
    default:
      return "Unknown message type";
  }
}

// @brief Describes TCP messages of the form:
// header        | message_type | count        | data  |
// ^- header_ptr   ^- body_ptr    ^- count_ptr   ^- data_ptr
//               | ---------------  body  ------------ |
// @param count number of elements of data
// @param size number of bytes of data in message. Must be a multiple of
// count
// MessageType::none indicates message may store data in the future, and so
// max_body_length bytes are allocated for the body.
class TCPMessage {
 public:
  enum { header_length = 15 };
  enum { max_body_length = 1000000 };
  enum { message_type_length = sizeof(MessageType) };
  enum { message_count_length = sizeof(size_t) };

  TCPMessage(const MessageType type)
      : m_type(type), m_count(0), m_data_size(0) {
    std::set<MessageType> request_types{
        MessageType::public_key_request, MessageType::relu_request,
        MessageType::public_key_ack,     MessageType::parameter_shape_request,
        MessageType::result_request,     MessageType::none};

    if (request_types.find(type) == request_types.end()) {
      throw std::invalid_argument("Request type not valid");
    }

    m_data = new char[header_length + max_body_length];

    encode_header();
    encode_message_type();
    encode_count();
    std::cout << "Done creating messge " << std::endl;
  }

  TCPMessage() : TCPMessage(MessageType::none) {}

  // Encodes message of count elements using data in stream
  TCPMessage(const MessageType type, size_t count,
             const std::stringstream& stream)
      : m_type(type), m_count(count) {
    const std::string& pk_str = stream.str();
    const char* pk_cstr = pk_str.c_str();
    m_data_size = pk_str.size();
    std::cout << "Creating message from stream, size " << m_data_size
              << std::endl;

    check_arguments();
    m_data = new char[header_length + max_body_length];

    encode_header();
    encode_message_type();
    encode_count();
    encode_data(pk_cstr);
  }

  void check_arguments() {
    if (m_count < 0) {
      throw std::invalid_argument("m_count must be non-negative");
    }
    if (m_count != 0 && m_data_size % m_count != 0) {
      std::cout << "size " << m_data_size << " count " << m_count << std::endl;
      throw std::invalid_argument("Size must be a multiple of count");
    }

    if (body_length() > max_body_length) {
      throw std::invalid_argument("Size " + std::to_string(body_length()) +
                                  " too large");
    }
  }

  TCPMessage(const MessageType type, const size_t count, const size_t size,
             const char* data)
      : m_type(type), m_count(count), m_data_size(size) {
    check_arguments();
    m_data = new char[header_length + max_body_length];
    std::cout << "Creating new message size " << header_length + max_body_length
              << std::endl;

    encode_header();
    std::cout << "Encoded header" << std::endl;
    encode_message_type();
    encode_count();
    encode_data(data);
    std::cout << "Done creating messge " << std::endl;
  }

  // copy assignment. TODO: enable as needed
  TCPMessage& operator=(const TCPMessage& other) = delete;

  // move assignment. TODO: enable as needed
  TCPMessage& operator=(TCPMessage&& other) = delete;

  // copy constructor
  TCPMessage(const TCPMessage& other) {
    m_type = other.m_type;
    m_count = other.m_count;
    m_data_size = other.m_data_size;

    m_data = new char[header_length + body_length()];
    std::memcpy(m_data, other.m_data, num_bytes());

    /*std::cout << "copy constructor" << std::endl;
    std::cout << "m_count " << m_count << std::endl;
    std::cout << "m_data_size " << m_data_size << std::endl;
    std::cout << "m_type " << message_type_to_string(m_type) << std::endl; */
  }

  // move constructor
  TCPMessage(TCPMessage&& other) noexcept
      : m_type(other.m_type),
        m_count(other.m_count),
        m_data_size(other.m_data_size) {
    m_data = new char[header_length + body_length()];
    std::memcpy(m_data, other.m_data, num_bytes());
  }

  ~TCPMessage() {
    // std::cout << "~TCPMessage() of type " << message_type_to_string(m_type)
    //          << std::endl;
    if (m_data) {
      // std::cout << "delete[] m_data" << std::endl;
      delete[] m_data;
    }
    // std::cout << "Done with ~TCPMessage() " << std::endl;
  }

  size_t count() { return m_count; }
  const size_t count() const { return m_count; }

  size_t element_size() {
    if (m_count == 0 || m_data_size % m_count != 0) {
      throw std::invalid_argument(
          "m_count == 0 or does not divide m_data_size");
    }
    return m_data_size / m_count;
  }
  const size_t element_size() const {
    if (m_count == 0 || m_data_size % m_count != 0) {
      throw std::invalid_argument(
          "m_count == 0 or does not divide m_data_size");
    }
    return m_data_size / m_count;
  }

  size_t num_bytes() { return header_length + body_length(); }
  const size_t num_bytes() const { return header_length + body_length(); }

  size_t data_size() { return m_data_size; }
  const size_t data_size() const { return m_data_size; }

  size_t body_length() const {
    return message_type_length + message_count_length + m_data_size;
  }

  MessageType message_type() { return m_type; }
  const MessageType message_type() const { return m_type; }

  char* header_ptr() { return m_data; }
  const char* header_ptr() const { return m_data; }

  char* body_ptr() { return header_ptr() + header_length; }
  const char* body_ptr() const { return header_ptr() + header_length; }

  char* count_ptr() { return body_ptr() + message_type_length; }
  const char* count_ptr() const { return body_ptr() + message_type_length; }

  char* data_ptr() { return count_ptr() + message_count_length; }
  const char* data_ptr() const { return count_ptr() + message_count_length; }

  // Given
  void encode_header() {
    char header[header_length + 1] = "";
    int to_encode = static_cast<int>(body_length());
    int ret = std::snprintf(header, sizeof(header), "%d", to_encode);
    if (ret < 0 || ret > sizeof(header)) {
      throw std::invalid_argument("Error encoding header");
    }
    std::cout << "Encoded header: body_length " << to_encode << std::endl;
    std::memcpy(m_data, header, header_length);
  }

  bool decode_header() {
    std::cout << "Decoding header " << std::endl;
    char header[header_length + 1] = "";
    std::strncat(header, m_data, header_length);
    size_t body_length = std::atoi(header);
    std::cout << "body_length " << body_length << std::endl;
    if (body_length > max_body_length) {
      std::cout << "Body length " << body_length << " too large" << std::endl;
      throw std::invalid_argument("Cannot decode header");
    }
    m_data_size = body_length - message_type_length - message_count_length;
    return true;
  }

  void encode_message_type() {
    std::memcpy(body_ptr(), &m_type, message_type_length);
    std::cout << "Encoded message type " << message_type_to_string(m_type)
              << std::endl;
  }

  void decode_message_type() {
    std::memcpy(&m_type, body_ptr(), message_type_length);
  }

  void encode_count() {
    std::memcpy(count_ptr(), &m_count, message_count_length);
    std::cout << "Encoded count " << m_count << std::endl;
  }

  void decode_count() {
    std::memcpy(&m_count, count_ptr(), message_count_length);
  }

  void encode_data(const char* data) {
    std::memcpy(data_ptr(), data, m_data_size);
    std::cout << "Encoded data size " << m_data_size << std::endl;
  }

  // Given m_data, parses to find m_datatype, m_count
  bool decode_body() {
    decode_message_type();
    decode_count();
    return true;
  }

 private:
  MessageType m_type;  // What data is being transmitted
  size_t m_count;      // Number of datatype in message
  size_t m_data_size;  // Nubmer of bytes in data part of message
  char* m_data;

};  // namespace he
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
