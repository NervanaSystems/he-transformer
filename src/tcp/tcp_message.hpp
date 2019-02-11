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
#include <memory>

namespace ngraph {
namespace runtime {
namespace he {
enum class Datatype { PUBLIC_KEY, CIPHERTEXT };
enum class Function { RELU, MAX, SOFTMAX, RESULT, NONE };
template <size_t PUBLIC_KEY_SIZE, size_t CIPHERSIZE>
class TCPMessage {
 public:
  // @brief Describes TCP messages of the form: (bytes, function, datatype,
  // count, data)
  TCPMessage(const Datatype datatype, const size_t count, Function function)
      : m_datatype(datatype), m_count(count), m_function(function) {
    bytes = m_header_length;
    bytes += sizeof(Function);
    bytes += sizeof(Datatype);

    switch (datatype) {
      case Datatype::PUBLIC_KEY:
        assert(count == 1);
        bytes += PUBLIC_KEY_SIZE;
        break;

      case Datatype::CIPHERTEXT:
        bytes += sizeof(count * CIPHERSIZE);
        break;
    };

    void* data_ptr = malloc(bytes);
    if (!data_ptr) {
      std::cout << "malloc failed to allocate memory of size " << bytes
                << std::endl;
      throw std::bad_alloc();
    }
  };

  ~TCPMessage() {
    if (data) {
      free(data);
    }
  }

  size_t bytes;         // How many bytes in the message
  Datatype m_datatype;  // What data is being transmitted
  size_t m_count;       // Number of datatype in message
  Function m_function;  // What function to compute on the data

  size_t m_header_length{sizeof(size_t)};  // Bits for the header.

  char* data;
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
