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
#include <string>

// TODO: better way to include log
#define PROJECT_ROOT_DIR "test_root_macro"

#include "ngraph/check.hpp"
#include "ngraph/log.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph {
namespace he {
enum class MessageType {
  none,
  encryption_parameters,
  eval_key,
  execute,
  max_request,
  max_result,
  minimum_request,
  minimum_result,
  parameter_shape_request,
  parameter_size,
  public_key,
  relu_request,
  relu_result,
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
    case MessageType::eval_key:
      return "eval_key";
      break;
    case MessageType::public_key:
      return "public_key";
      break;
    case MessageType::execute:
      return "execute";
      break;
    case MessageType::minimum_request:
      return "minimum_request";
      break;
    case MessageType::minimum_result:
      return "minimum_result";
      break;
    case MessageType::max_request:
      return "max_request";
      break;
    case MessageType::max_result:
      return "max_result";
      break;
    case MessageType::parameter_size:
      return "parameter_size";
      break;
    case MessageType::parameter_shape_request:
      return "parameter_shape_request";
      break;
    case MessageType::relu_result:
      return "relu_result";
      break;
    case MessageType::relu_request:
      return "relu_request";
      break;
    case MessageType::result:
      return "result";
      break;
    case MessageType::result_request:
      return "result_request";
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
class TCPMessage {
 public:
  enum { header_length = 15 };
  enum { max_body_length = 12000000000UL };
  enum { message_type_length = sizeof(MessageType) };
  enum { message_count_length = sizeof(size_t) };

  // Creates message with data buffer large enough to store max_body_length
  // Note: this requires a lot of memory and should be avoided where possible
  // TODO: more scalable solution
  TCPMessage(const MessageType type)
      : m_type(type), m_count(0), m_data_size(0) {
    std::set<MessageType> request_types{
        MessageType::relu_request, MessageType::parameter_shape_request,
        MessageType::result_request, MessageType::none};

    if (request_types.find(type) == request_types.end()) {
      throw std::invalid_argument("Request type not valid");
    }
    check_arguments();
    // TODO: use malloc
    m_data = new char[header_length + max_body_length];
    encode_header();
    encode_message_type();
    encode_count();
  }

  TCPMessage() : TCPMessage(MessageType::none) {}

  // Encodes message of count elements using data in stream
  TCPMessage(const MessageType type, size_t count, std::stringstream&& stream)
      : m_type(type), m_count(count) {
    stream.seekp(0, std::ios::end);
    m_data_size = stream.tellp();
    NGRAPH_INFO << "m_data_size stream " << m_data_size;

    check_arguments();
    // TODO: use malloc
    m_data = new char[header_length + body_length()];
    encode_header();
    encode_message_type();
    encode_count();
    encode_data(std::move(stream));
  }

  TCPMessage(const MessageType type,
             const std::vector<seal::Ciphertext>& ciphers)
      : m_type(type), m_count(ciphers.size()) {
    NGRAPH_INFO << "Creating message from " << ciphers.size()
                << " seal ciphertexts";

    std::stringstream first;
    ciphers[0].save(first);
    first.seekp(0, std::ios::end);
    size_t first_cipher_size = first.tellp();
    NGRAPH_INFO << "first_cipher_size " << first_cipher_size;

    m_data_size = first_cipher_size * m_count;
    NGRAPH_INFO << "m_data_size " << m_data_size;

    check_arguments();
    // TODO: use malloc
    m_data = new char[header_length + body_length()];
    encode_header();
    encode_message_type();
    encode_count();

#pragma omp parallel for
    for (size_t i = 0; i < ciphers.size(); ++i) {
      size_t offset = i * first_cipher_size;
      std::stringstream ss;
      // TODO: save directly to buffer
      ciphers[i].save(ss);
      ss.seekp(0, std::ios::end);
      size_t cipher_size = ss.tellp();
      NGRAPH_CHECK(cipher_size == first_cipher_size, "cipher size ",
                   cipher_size, "doesn't match first", first_cipher_size);

      std::stringbuf* pbuf = ss.rdbuf();
      pbuf->sgetn(data_ptr() + offset, first_cipher_size);
    }
    NGRAPH_INFO << "first_cipher_size " << first_cipher_size;
  }

  TCPMessage(const MessageType type, const size_t count, const size_t size,
             const char* data)
      : m_type(type), m_count(count), m_data_size(size) {
    check_arguments();
    // TODO: use malloc
    m_data = new char[header_length + body_length()];
    NGRAPH_INFO << "TCPMessage from pointer";
    encode_header();
    encode_message_type();
    encode_count();
    encode_data(data);
  }

  TCPMessage& operator=(TCPMessage&& other) {
    if (this != &other) {
      m_data = other.m_data;
      m_type = other.m_type;
      m_count = other.m_count;
      m_data_size = other.m_data_size;
      other.m_data = nullptr;
      other.m_data_size = 0;
      other.m_count = 0;
      other.m_type = MessageType::none;
    }
    return *this;
  };

  TCPMessage(TCPMessage&& other)
      : m_data(other.m_data),
        m_type(other.m_type),
        m_count(other.m_count),
        m_data_size(other.m_data_size) {
    other.m_data = nullptr;
  };

  TCPMessage& operator=(const TCPMessage&) = delete;
  TCPMessage(const TCPMessage& other) = delete;

  ~TCPMessage() {
    if (m_data != nullptr) {
      delete[] m_data;
    }
  }

  void check_arguments() {
    if (m_count < 0) {
      throw std::invalid_argument("m_count must be non-negative");
    }
    if (m_count != 0 && m_data_size % m_count != 0) {
      NGRAPH_INFO << "Error: size " << m_data_size
                  << " not a multiple of count " << m_count;
      throw std::invalid_argument("Size must be a multiple of count");
    }

    if (body_length() > max_body_length) {
      throw std::invalid_argument("Size " + std::to_string(body_length()) +
                                  " too large");
    }
  }

  size_t count() { return m_count; }
  const size_t count() const { return m_count; }

  size_t element_size() {
    if (m_count == 0) {
      throw std::invalid_argument("m_count == 0");
    }
    if (m_data_size % m_count != 0) {
      std::stringstream ss;
      ss << "m_count " << m_count << " does not divide m_data_size "
         << m_data_size;
      throw std::invalid_argument(ss.str());
    }
    return m_data_size / m_count;
  }

  const size_t element_size() const {
    if (m_count == 0) {
      throw std::invalid_argument("m_count == 0");
    }
    if (m_data_size % m_count != 0) {
      std::stringstream ss;
      ss << "m_count " << m_count << " does not divide m_data_size "
         << m_data_size;
      throw std::invalid_argument(ss.str());
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
    size_t to_encode = body_length();
    int ret = std::snprintf(header, sizeof(header), "%zu", to_encode);
    if (ret < 0 || (size_t)ret > sizeof(header)) {
      throw std::invalid_argument("Error encoding header");
    }
    std::memcpy(m_data, header, header_length);
  }

  bool decode_header() {
    char header[header_length + 1] = "";
    std::strncat(header, m_data, header_length);
    std::string header_str(header);
    std::stringstream sstream(header_str);
    size_t body_length;
    sstream >> body_length;
    if (body_length > max_body_length) {
      NGRAPH_INFO << "Body length " << body_length << " too large";
      throw std::invalid_argument("Cannot decode header");
    }
    m_data_size = body_length - message_type_length - message_count_length;
    return true;
  }

  void encode_message_type() {
    std::memcpy(body_ptr(), &m_type, message_type_length);
  }

  void decode_message_type() {
    std::memcpy(&m_type, body_ptr(), message_type_length);
  }

  void encode_count() {
    std::memcpy(count_ptr(), &m_count, message_count_length);
  }

  void decode_count() {
    std::memcpy(&m_count, count_ptr(), message_count_length);
  }

  void encode_data(const char* data) {
    NGRAPH_INFO << "Encoding data from char* size " << m_data_size;
    std::memcpy(data_ptr(), data, m_data_size);
    NGRAPH_INFO << "done with memcpy ";
  }

  void encode_data(const std::stringstream&& data) {
    NGRAPH_INFO << "Encoding data from stream size " << m_data_size;
    std::stringbuf* pbuf = data.rdbuf();
    pbuf->sgetn(data_ptr(), m_data_size);
    // const void* data_buf = (const void*)(data.rdbuf()->data());
    // std::memcpy(data_ptr(), data_buf, m_data_size);
    NGRAPH_INFO << "done with memcpy ";
  }

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
};
}  // namespace he
}  // namespace ngraph
