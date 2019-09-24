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

#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph {
namespace he {
enum class HETensorTypeInfo { unknown = 0, cipher = 1, plain = 2 };

class HESealBackend;
class HETensor : public runtime::Tensor {
 public:
  HETensor(const element::Type& element_type, const Shape& shape,
           const bool packed = false, const std::string& name = "external");
  virtual ~HETensor() override {}

  /// @brief Write bytes directly into the tensor
  /// @param p Pointer to source of data
  /// element-aligned.
  /// @param n Number of bytes to write, must be integral number of elements.
  virtual void write(const void* p, size_t n) override = 0;

  /// @brief Read bytes directly from the tensor
  /// @param p Pointer to destination for data
  /// element-aligned.
  /// @param n Number of bytes to read, must be integral number of elements.
  virtual void read(void* p, size_t n) const override = 0;

  /// @brief Reduces shape along batch axis
  /// @param shape Input shape to batch
  /// @param batch_axis Axis along which to batch
  /// @return Shape after batching along batch axis
  static Shape pack_shape(const Shape& shape, size_t batch_axis = 0);

  /// @brief Returns the shape of the un-expanded (i.e. packed) tensor.
  const Shape& get_packed_shape() const { return m_packed_shape; }

  /// @brief Returns the shape of the expanded (batched) tensor.
  const Shape& get_expanded_shape() const { return get_shape(); }

  inline size_t get_batch_size() { return m_batch_size; }

  inline size_t get_batched_element_count() {
    return get_element_count() / get_batch_size();
  }

  inline bool is_packed() { return m_packed; }

  virtual const HETensorTypeInfo& get_type_info() const { return type_info; }

  template <typename HETensorType>
  bool is_type() const {
    return &get_type_info() == &HETensorType::type_info;
  }

  static constexpr HETensorTypeInfo type_info{HETensorTypeInfo::unknown};

 protected:
  void check_io_bounds(const void* p, size_t n) const;
  const HESealBackend& m_he_seal_backend;
  bool m_packed;        // Whether or not the tensor is packed, i.e. stores more
                        // than one scalar per element.
  size_t m_batch_size;  // If m_packed, corresponds to first shape dimesion.
  Shape m_packed_shape;
};

template <typename HETensorType>
std::shared_ptr<HETensorType> he_tensor_as_type(
    const std::shared_ptr<HETensor>& he_tensor) {
  NGRAPH_CHECK(he_tensor->is_type<HETensorType>(), "incorrect tensor type");
  return std::static_pointer_cast<HETensorType>(he_tensor);
}

}  // namespace he
}  // namespace ngraph
