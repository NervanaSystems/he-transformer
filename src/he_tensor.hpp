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

#include "he_plaintext.hpp"
#include "he_type.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph {
namespace he {

class HESealBackend;
/// \brief Class representing a Tensor of either ciphertexts or plaintexts
class HETensor : public runtime::Tensor {
 public:
  /// \brief Constructs a generic HETensor
  /// \param[in] element_type Datatype of underlying data
  /// \param[in] shape Shape of tensor
  /// \param[in] packed Whether or not to use plaintext packing
  /// \param[in] name Name of the tensor
  HETensor(const element::Type& element_type, const Shape& shape,
           const HESealBackend& he_seal_backend,
           const std::string& name = "external");

  virtual ~HETensor() override {}

  /// \brief Write bytes directly into the tensor
  /// \param[in] p Pointer to source of data
  /// \param[in] n Number of bytes to write, must be integral number of elements
  void write(const void* p, size_t n) override {
    throw ngraph_error("Write unimplemented");
  };

  /// \brief Read bytes directly from the tensor
  /// \param[out] p Pointer to destination for data
  /// \param[in] n Number of bytes to read, must be integral number of elements.
  void read(void* p, size_t n) const override {
    throw ngraph_error("Read unimplemented");
  }

  /// \brief Reduces shape along pack axis
  /// \param[in] shape Input shape to pack
  /// \param[in] pack_axis Axis along which to pack
  /// \return Shape after packing along pack axis
  static Shape pack_shape(const Shape& shape, size_t pack_axis = 0);

  /// \brief Expands shape along pack axis
  /// \param[in] shape Input shape to pack
  /// \param[in] pack_size New size of pack axis
  /// \param[in] pack_axis Axis along which to pack
  /// \return Shape after expanding along pack axis
  static Shape unpack_shape(const Shape& shape, size_t pack_size,
                            size_t pack_axis = 0);

  const std::vector<HEType>& data() const { return m_data; }

  std::vector<HEType>& data() { return m_data; }

  /// \brief Returns the batch size of a given shape
  /// \param[in] shape Shape of the tensor
  /// \param[in] packed Whether or not batch-axis packing is used
  static uint64_t batch_size(const Shape& shape, const bool packed);

  /// \brief Returns the shape of the un-expanded (i.e. packed) tensor.
  const Shape& get_packed_shape() const { return m_packed_shape; }

  /// \brief Returns the shape of the expanded tensor.
  const Shape& get_expanded_shape() const { return get_shape(); }

  /// \brief Returns packing factor used in the tensor
  inline size_t get_batch_size() const { return m_batch_size; }

  /// \brief Returns number of ciphertext / plaintext objects in the tensor
  inline size_t get_batched_element_count() const {
    return get_element_count() / get_batch_size();
  }

  /// \brief Returns whether or not the tensor is packed
  inline bool is_packed() const { return m_packed; }

 private:
  std::vector<HEType> m_data;
  const HESealBackend& m_he_seal_backend;

  void check_io_bounds(const void* p, size_t n) const;
  bool m_packed;
  size_t m_batch_size;
  Shape m_packed_shape;
};

}  // namespace he
}  // namespace ngraph
