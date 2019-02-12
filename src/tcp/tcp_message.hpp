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

namespace ngraph {
namespace runtime {
namespace he {
enum class Datatype { PUBLIC_KEY, CIPHERTEXT };
enum class MPCFunction { RELU, MAX, SOFTMAX, RESULT, NONE };

class TCPMessage {
 public:
  enum { header_length = sizeof(size_t) };
  enum { max_body_length = 1000 };

  TCPMessage()
      : m_bytes(0),
        m_body_length(0),
        m_datatype(Datatype::PUBLIC_KEY),
        m_count(0),
        m_function(MPCFunction::NONE) {}

  // @brief Describes TCP messages of the form: (bytes, function,
  // datatype, count, data)
  // @oaram size size of incoming data. Must be a multiple of count
  TCPMessage(const Datatype datatype, const size_t count, const size_t size,
             const MPCFunction function, const char* data)
      : m_datatype(datatype), m_count(count), m_function(function) {
    if (size % count != 0) {
      throw std::invalid_argument("Size must be a multiple of count");
    }

    m_bytes = header_length;
    m_bytes += sizeof(MPCFunction);
    m_bytes += sizeof(Datatype);
    m_bytes += sizeof(size);

    m_data_ptr = malloc(m_bytes);
    if (!m_data_ptr) {
      std::cout << "malloc failed to allocate memory of size " << m_bytes
                << std::endl;
      throw std::bad_alloc();
    }

    // Copy header
    std::cout << "bytes " << m_bytes << std::endl;
    std::cout << "copying " << sizeof(m_bytes) << std::endl;
    std::memcpy(m_data_ptr, &m_bytes, sizeof(m_bytes));
    std::cout << "Done creating TCP message" << std::endl;
  }

  void* data() const { return m_data_ptr; }

  bool decode_header() {
    char header[header_length + 1] = "";
    std::strncat(header, (char*)m_data_ptr, header_length);
    m_body_length = std::atoi(header);
    if (m_body_length > max_body_length) {
      std::cout << "Message body length " << m_body_length
                << " exceeds maximum " << max_body_length << std::endl;
      m_body_length = 0;
      return false;
    }
    return true;
  }

  ~TCPMessage() {
    if (m_data_ptr) {
      free(m_data_ptr);
    }
  }

  size_t size() const { return m_bytes; }

  size_t m_bytes;          // How many bytes in the message
  size_t m_body_length;    // How many bytes in message body
  Datatype m_datatype;     // What data is being transmitted
  size_t m_count;          // Number of datatype in message
  MPCFunction m_function;  // What function to compute on the data

  void* m_data_ptr;
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
