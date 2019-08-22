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

#include "message.pb.h"
#include "ngraph/check.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace he {

class NewTCPMessage {
 public:
  enum { header_length = sizeof(size_t) };
  using data_buffer = std::vector<char>;

  NewTCPMessage() = default;

  NewTCPMessage(he_proto::TCPMessage& proto_message)
      : m_proto_message(std::make_shared<he_proto::TCPMessage>(proto_message)) {
  }

  NewTCPMessage(std::shared_ptr<he_proto::TCPMessage> proto_message)
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

    // TODO: remove
    size_t decoded_size = decode_header(buffer);
    NGRAPH_INFO << "Decoded " << decoded_size;
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

enum class MessageType {
  none,
  encryption_parameters,
  eval_key,
  execute,
  maxpool_request,
  maxpool_result,
  minimum_request,
  minimum_result,
  parameter_shape_request,
  parameter_size,
  public_key,
  relu_request,
  relu6_request,
  relu_result,
  result,
  result_request
};

inline std::ostream& operator<<(std::ostream& os, const MessageType& type) {
  switch (type) {
    case MessageType::none:
      os << "none";
      break;
    case MessageType::encryption_parameters:
      os << "encryption_parameters";
      break;
    case MessageType::eval_key:
      os << "eval_key";
      break;
    case MessageType::public_key:
      os << "public_key";
      break;
    case MessageType::execute:
      os << "execute";
      break;
    case MessageType::minimum_request:
      os << "minimum_request";
      break;
    case MessageType::minimum_result:
      os << "minimum_result";
      break;
    case MessageType::maxpool_request:
      os << "maxpool_request";
      break;
    case MessageType::maxpool_result:
      os << "maxpool_result";
      break;
    case MessageType::parameter_size:
      os << "parameter_size";
      break;
    case MessageType::parameter_shape_request:
      os << "parameter_shape_request";
      break;
    case MessageType::relu_result:
      os << "relu_result";
      break;
    case MessageType::relu_request:
      os << "relu_request";
      break;
    case MessageType::relu6_request:
      os << "relu6_request";
      break;
    case MessageType::result:
      os << "result";
      break;
    case MessageType::result_request:
      os << "result_request";
      break;
    default:
      os << "Unknown message type";
  }
  return os;
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
  enum { max_body_length = 39900000000UL };
  enum { default_body_length = 400000000UL };
  enum { message_type_length = sizeof(MessageType) };
  enum { message_count_length = sizeof(size_t) };

  // Creates message with data buffer large enough to store
  // default_body_length
  TCPMessage(const MessageType type)
      : m_type(type), m_count(0), m_data_size(0) {
    std::set<MessageType> request_types{
        MessageType::relu_request, MessageType::parameter_shape_request,
        MessageType::result_request, MessageType::none};

    if (request_types.find(type) == request_types.end()) {
      throw std::invalid_argument("Request type not valid");
    }
    check_arguments();
    size_t body_malloc_size =
        std::max(body_length(), static_cast<size_t>(default_body_length));
    m_data =
        static_cast<char*>(ngraph_malloc(header_length + body_malloc_size));
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

    check_arguments();
    size_t body_malloc_size =
        std::max(body_length(), static_cast<size_t>(default_body_length));
    m_data =
        static_cast<char*>(ngraph_malloc(header_length + body_malloc_size));
    encode_header();
    encode_message_type();
    encode_count();
    encode_data(std::move(stream));
  }

  TCPMessage(const MessageType type, const seal::Ciphertext& cipher)
      : TCPMessage(type, std::vector<seal::Ciphertext>{cipher}) {}

  TCPMessage(const MessageType type,
             const std::vector<std::shared_ptr<SealCiphertextWrapper>>& ciphers)
      : m_type(type), m_count(ciphers.size()) {
    NGRAPH_CHECK(ciphers.size() > 0, "No ciphertexts in TCPMessage");
    size_t cipher_size = ciphertext_size(ciphers[0]->ciphertext());
    m_data_size = cipher_size * m_count;

    check_arguments();
    size_t body_malloc_size =
        std::max(body_length(), static_cast<size_t>(default_body_length));
    m_data =
        static_cast<char*>(ngraph_malloc(header_length + body_malloc_size));
    encode_header();
    encode_message_type();
    encode_count();

#pragma omp parallel for
    for (size_t i = 0; i < ciphers.size(); ++i) {
      size_t offset = i * cipher_size;
      save_cipher_to_message(ciphers[i]->ciphertext(), offset);
      NGRAPH_CHECK(ciphertext_size(ciphers[i]->ciphertext()) == cipher_size,
                   "Cipher sizes don't match. Got size ",
                   ciphertext_size(ciphers[i]->ciphertext()), ", expected ",
                   cipher_size);
    }
  }

  TCPMessage(const MessageType type,
             const std::vector<seal::Ciphertext>& ciphers)
      : m_type(type), m_count(ciphers.size()) {
    NGRAPH_CHECK(ciphers.size() > 0, "No ciphertexts in TCPMessage");
    size_t cipher_size = ciphertext_size(ciphers[0]);
    m_data_size = cipher_size * m_count;

    check_arguments();
    size_t body_malloc_size =
        std::max(body_length(), static_cast<size_t>(default_body_length));
    m_data =
        static_cast<char*>(ngraph_malloc(header_length + body_malloc_size));
    encode_header();
    encode_message_type();
    encode_count();

#pragma omp parallel for
    for (size_t i = 0; i < ciphers.size(); ++i) {
      size_t offset = i * cipher_size;
      save_cipher_to_message(ciphers[i], offset);
      NGRAPH_CHECK(ciphertext_size(ciphers[i]) == cipher_size,
                   "Cipher sizes don't match. Got size ",
                   ciphertext_size(ciphers[i]), ", expected ", cipher_size);
    }
  }

  TCPMessage(const MessageType type, const size_t count, const size_t size,
             const char* data)
      : m_type(type), m_count(count), m_data_size(size) {
    check_arguments();
    size_t body_malloc_size =
        std::max(body_length(), static_cast<size_t>(default_body_length));
    m_data =
        static_cast<char*>(ngraph_malloc(header_length + body_malloc_size));
    encode_header();
    encode_message_type();
    encode_count();
    encode_data(data);
  }

  TCPMessage& operator=(TCPMessage&& other) {
    if (this != &other) {
      m_type = other.m_type;
      m_count = other.m_count;
      m_data_size = other.m_data_size;
      m_data = other.m_data;
      other.m_data = nullptr;
      other.m_data_size = 0;
      other.m_count = 0;
      other.m_type = MessageType::none;
    }
    return *this;
  }

  TCPMessage(TCPMessage&& other)
      : m_type(other.m_type),
        m_count(other.m_count),
        m_data_size(other.m_data_size),
        m_data(other.m_data) {
    other.m_data = nullptr;
  }

  TCPMessage& operator=(const TCPMessage&) = delete;
  TCPMessage(const TCPMessage& other) = delete;

  ~TCPMessage() { ngraph_free(m_data); }

  void check_arguments() {
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

  size_t count() const { return m_count; }

  size_t element_size() const {
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

  size_t num_bytes() const { return header_length + body_length(); }

  size_t data_size() const { return m_data_size; }

  size_t body_length() const {
    return message_type_length + message_count_length + m_data_size;
  }

  MessageType message_type() const { return m_type; }

  char* header_ptr() { return m_data; }
  const char* header_ptr() const { return m_data; }

  char* body_ptr() { return header_ptr() + header_length; }
  const char* body_ptr() const { return header_ptr() + header_length; }

  char* count_ptr() { return body_ptr() + message_type_length; }
  const char* count_ptr() const { return body_ptr() + message_type_length; }

  char* data_ptr() { return count_ptr() + message_count_length; }
  const char* data_ptr() const { return count_ptr() + message_count_length; }

  void encode_header() {
    char header[header_length + 1] = "";
    size_t to_encode = body_length();
    int ret = std::snprintf(header, sizeof(header), "%zu", to_encode);
    if (ret < 0 || static_cast<size_t>(ret) > sizeof(header)) {
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

    // Resize to fit message
    if (body_length > default_body_length) {
      ngraph_free(m_data);
      m_data = static_cast<char*>(ngraph_malloc(header_length + body_length));
      encode_header();
    }
    return true;
  }

  inline void encode_message_type() {
    std::memcpy(body_ptr(), &m_type, message_type_length);
  }

  inline void decode_message_type() {
    std::memcpy(&m_type, body_ptr(), message_type_length);
  }

  inline void encode_count() {
    std::memcpy(count_ptr(), &m_count, message_count_length);
  }

  inline void decode_count() {
    std::memcpy(&m_count, count_ptr(), message_count_length);
  }

  inline void encode_data(const char* data) {
    std::memcpy(data_ptr(), data, m_data_size);
  }

  inline void encode_data(const std::stringstream&& data) {
    std::stringbuf* pbuf = data.rdbuf();
    pbuf->sgetn(data_ptr(), m_data_size);
  }

  inline bool decode_body() {
    decode_message_type();
    decode_count();
    return true;
  }

  inline void load_cipher(seal::Ciphertext& cipher, size_t index,
                          std::shared_ptr<seal::SEALContext> context) const {
    NGRAPH_CHECK(index < count(), "Index too large");
    ngraph::he::load(cipher, context,
                     static_cast<void*>(const_cast<char*>(
                         data_ptr() + index * element_size())));
  }

  inline void save_cipher_to_message(const seal::Ciphertext& cipher,
                                     size_t offset) {
    ngraph::he::save(cipher, data_ptr() + offset);
  }

 private:
  MessageType m_type;  // What data is being transmitted
  size_t m_count;      // Number of datatype in message
  size_t m_data_size;  // Nubmer of bytes in data part of message
  char* m_data;
};
}  // namespace he
}  // namespace ngraph