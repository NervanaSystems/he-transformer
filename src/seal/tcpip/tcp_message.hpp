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
#include <unordered_map>

namespace ngraph {
namespace runtime {
namespace he {
enum class DataType { PUBLICKEY, CIPHERTEXT };
enum class Function { RELU, MAX, SOFTMAX, RESULT, NONE };
template <size_t PKSIZE, size_t CIPHERSIZE>
class TCPMessage {
 public:
  // @brief Describes TCP messages of the form: (bytes, function, datatype,
  // count, data)
  TCPMessage(const DataType datatype, const size_t count, Function function)
      : m_datatype(datatype), m_count(count), m_function(function) {
    bytes = header_size;
    bytes += sizeof(Function);
    bytes += sizeof(DataType);

    switch (datatype) {
      case PUBLICKEY:
        bytes += sizeof(count * PKSIZE);
        break;

      case CIPHERTEXT:
        bytes += sizeof(count * CIPHERSIZE);
        break;
    };

    void* data_ptr = malloc(bytes);
    if (!data_ptr) {
      std::cout << "malloc failed to allocate memory of size " << size
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
  DataType m_datatype;  // What data is being transmitted
  size_t m_count;       // Number of datatype in message
  Function m_function;  // What function to compute on the data

  size_t header_length = sizeof(size_t);  // Bits for the header.

  char* data;
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
