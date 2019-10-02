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
/// \brief Class representing a tensor of plaintext values
class HEPlainTensor : public HETensor {
 public:
  /// \brief Construct a plain tensor
  /// \param[in] element_type Data type of elements in the tensor
  /// \param[in] shape Shape of the tensor
  /// \param[in] packed Whether or not to use plaintext packing
  /// \param[in] name Name of the tensor
  HEPlainTensor(const element::Type& element_type, const Shape& shape,
                const bool packed = false,
                const std::string& name = "external");

  /// \brief Write bytes directly into the tensor after encoding
  /// \param[in] source Pointer to source of data
  /// \param[in] n Number of bytes to write, must be integral number of
  /// elements.
  void write(const void* source, size_t n) override;

  /// \brief Read bytes directly from the tensor after decoding
  /// \param[in] target Pointer to destination for data
  /// \param[in] n Number of bytes to read, must be integral number of elements.
  void read(void* target, size_t n) const override;

  /// \brief Returns the plaintexts in the tensor
  inline std::vector<ngraph::he::HEPlaintext>& get_elements() {
    return m_plaintexts;
  }

  /// \brief Returns the plaintext at a given index
  /// \param[in] index Index from which to return the plaintext
  inline ngraph::he::HEPlaintext& get_element(size_t index) {
    return m_plaintexts[index];
  }

  /// \brief Clears the plaintexts
  inline void reset() { m_plaintexts.clear(); }

  /// \brief Returns the number of plaintexts in the tensor
  inline size_t num_plaintexts() { return m_plaintexts.size(); }

  /// \brief Sets the tensor to the given plaintexts
  /// \throws ngraph_error if wrong number of elements are used
  /// \param[in] elements Plaintexts to set the tensor to
  void set_elements(const std::vector<ngraph::he::HEPlaintext>& elements);

  /// \brief Returns type information about the tensor
  /// /returns a reference to a HETensorTypeInfo object
  const HETensorTypeInfo& get_type_info() const override { return type_info; }

  /// \brief Represents a HEPlainTensor type
  static constexpr HETensorTypeInfo type_info{HETensorTypeInfo::plain};

  /// \brief Writes plaintexts to vector of protobufs
  /// Due to 2GB limit in protobuf, the cipher tensor may be spread across
  /// multiple protobuf messages.
  /// \param[out] protos Target to write plaintexts to
  /// \param[in] name Name of the tensor to save
  inline void save_to_proto(std::vector<he_proto::PlainTensor>& protos,
                            const std::string& name = "") {
    // TODO: support large shapes
    protos.resize(1);
    protos[0].set_name(name);
    protos[0].set_packed(is_packed());

    std::vector<uint64_t> int_shape{get_shape()};
    *protos[0].mutable_shape() = {int_shape.begin(), int_shape.end()};

    protos[0].set_offset(0);

    for (const auto& plaintext : m_plaintexts) {
      plaintext.save(*protos[0].add_plaintexts());
    }
  }

 private:
  std::vector<ngraph::he::HEPlaintext> m_plaintexts;
};
}  // namespace he
}  // namespace ngraph
