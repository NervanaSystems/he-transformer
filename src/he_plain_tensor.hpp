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
  /// @param source Pointer to source of data
  /// @param n Number of bytes to write, must be integral number of elements.
  void write(const void* source, size_t n) override;

  /// @brief Read bytes directly from the tensor after decoding
  /// @param target Pointer to destination for data
  /// @param n Number of bytes to read, must be integral number of elements.
  void read(void* target, size_t n) const override;

  inline std::vector<ngraph::he::HEPlaintext>& get_elements() {
    return m_plaintexts;
  }

  inline ngraph::he::HEPlaintext& get_element(size_t i) {
    return m_plaintexts[i];
  }

  static inline std::vector<double> type_vec_to_double_vec(
      const void* src, const element::Type& element_type, size_t n) {
    std::vector<double> ret(n);
    char* src_with_offset = static_cast<char*>(const_cast<void*>(src));
    for (size_t i = 0; i < n; ++i) {
      ret[i] = type_to_double(src_with_offset, element_type);
      ++src_with_offset;
    }
    return ret;
  }

  static inline double type_to_double(const void* src,
                                      const element::Type& element_type) {
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
    switch (element_type.get_type_enum()) {
      case element::Type_t::f32:
        return static_cast<double>(*static_cast<const float*>(src));
        break;
      case element::Type_t::f64:
        return static_cast<double>(*static_cast<const double*>(src));
        break;
      case element::Type_t::i8:
      case element::Type_t::i16:
      case element::Type_t::i32:
      case element::Type_t::i64:
      case element::Type_t::u8:
      case element::Type_t::u16:
      case element::Type_t::u32:
      case element::Type_t::u64:
      case element::Type_t::dynamic:
      case element::Type_t::undefined:
      case element::Type_t::bf16:
      case element::Type_t::f16:
      case element::Type_t::boolean:
        NGRAPH_CHECK(false, "Unsupported element type", element_type);
        break;
    }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
  }

  static constexpr size_t internal_type_byte_size = HEPlaintext::type_byte_size;

  inline void reset() { m_plaintexts.clear(); }

  inline size_t num_plaintexts() { return m_plaintexts.size(); }

  void set_elements(const std::vector<ngraph::he::HEPlaintext>& elements);

 private:
  std::vector<ngraph::he::HEPlaintext> m_plaintexts;
  size_t m_num_elements;
};
}  // namespace he
}  // namespace ngraph
