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
#include <string>
#include <vector>

#include "he_ciphertext.hpp"
#include "he_tensor.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph {
namespace runtime {
namespace he {
class HECipherTensor : public HETensor {
 public:
  HECipherTensor(const element::Type& element_type, const Shape& shape,
                 const HEBackend* he_backend,
                 const std::shared_ptr<HECiphertext> he_ciphertext,
                 const bool batched = false,
                 const std::string& name = "external");

  /// @brief Write bytes directly into the tensor after encoding and encrypting
  /// @param p Pointer to source of data
  /// @param tensor_offset Offset (bytes) into tensor storage to begin writing.
  ///        Must be element-aligned.
  /// @param n Number of bytes to write, must be integral number of elements.
  void write(const void* p, size_t tensor_offset, size_t n) override;

  /// @brief Read bytes directly from the tensor after decrypting and decoding
  /// @param p Pointer to destination for data
  /// @param tensor_offset Offset (bytes) into tensor storage to begin reading.
  ///        Must be element-aligned.
  /// @param n Number of bytes to read, must be integral number of elements.
  void read(void* target, size_t tensor_offset, size_t n) const override;

  void set_elements(
      const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& elements);

  inline std::vector<std::shared_ptr<runtime::he::HECiphertext>>&
  get_elements() noexcept {
    return m_cipher_texts;
  }

  inline std::shared_ptr<runtime::he::HECiphertext>& get_element(size_t i) {
    return m_cipher_texts[i];
  }

 private:
  std::vector<std::shared_ptr<runtime::he::HECiphertext>> m_cipher_texts;
  size_t m_num_elements;
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
