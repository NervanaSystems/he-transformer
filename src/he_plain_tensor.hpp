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
                const HESealBackend& he_seal_backend,
                const bool batched = false,
                const std::string& name = "external");

  /// @brief Write bytes directly into the tensor after encoding
  /// @param p Pointer to source of data
  /// @param n Number of bytes to write, must be integral number of elements.
  void write(const void* source, size_t n) override;

  /// @brief Read bytes directly from the tensor after decoding
  /// @param p Pointer to destination for data
  /// @param n Number of bytes to read, must be integral number of elements.
  void read(void* target, size_t n) const override;

  inline std::vector<ngraph::he::HEPlaintext>& get_elements() {
    return m_plaintexts;
  }

  inline ngraph::he::HEPlaintext& get_element(size_t i) {
    return m_plaintexts[i];
  }

  inline void reset() { m_plaintexts.clear(); }

  inline size_t num_plaintexts() { return m_plaintexts.size(); }

  void set_elements(const std::vector<ngraph::he::HEPlaintext>& elements);

 private:
  std::vector<ngraph::he::HEPlaintext> m_plaintexts;
  size_t m_num_elements;
};
}  // namespace he
}  // namespace ngraph
