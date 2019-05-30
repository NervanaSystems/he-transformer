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

#include "he_plaintext.hpp"
#include "he_tensor.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph {
namespace he {
class HEPlainTensor : public HETensor {
 public:
  HEPlainTensor(const element::Type& element_type, const Shape& shape,
                const HEBackend* he_backend,
                const std::unique_ptr<ngraph::he::HEPlaintext> he_plaintext,
                const bool batched = false,
                const std::string& name = "external");

  /// @brief Write bytes directly into the tensor after encoding
  /// @param p Pointer to source of data
  /// @param tensor_offset Offset (bytes) into tensor storage to begin writing.
  ///        Must be element-aligned.
  /// @param n Number of bytes to write, must be integral number of elements.
  void write(const void* source, size_t tensor_offset, size_t n) override;

  /// @brief Read bytes directly from the tensor after decoding
  /// @param p Pointer to destination for data
  /// @param tensor_offset Offset (bytes) into tensor storage to begin reading.
  ///        Must be element-aligned.
  /// @param n Number of bytes to read, must be integral number of elements.
  void read(void* target, size_t tensor_offset, size_t n) const override;

  inline std::vector<std::unique_ptr<ngraph::he::HEPlaintext>>&
  get_elements() noexcept {
    return m_plaintexts;
  }

  inline std::unique_ptr<ngraph::he::HEPlaintext>& get_element(size_t i) {
    return m_plaintexts[i];
  }

 private:
  std::vector<std::unique_ptr<ngraph::he::HEPlaintext>> m_plaintexts;
  size_t m_num_elements;
};
}  // namespace he
}  // namespace ngraph
