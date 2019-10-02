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

#include <vector>

#include "protos/message.pb.h"

namespace ngraph {
namespace he {
/// \brief Class representing a plaintext value
class HEPlaintext {
 public:
  /// \brief Constructs an empty plaintext value
  HEPlaintext() = default;

  /// \brief Constructs a plaintext from the given values
  /// \param[in] values Values stored in the plaintext
  HEPlaintext(const std::vector<double>& values) : m_values(values) {}

  /// \brief Constructs a plaintext storing a single value
  /// \param[in] value Value stored in the plaintext
  HEPlaintext(const double value) : m_values{std::vector<double>{value}} {}

  /// \brief Returns a reference to the stored values
  inline const std::vector<double>& values() const { return m_values; }

  /// \brief Returns the first value stored in the plaintext
  inline double first_value() const { return m_values[0]; }

  /// \brief Sets the plaintext to store a single value
  /// \param[in] value Value to store in the plaintext
  inline void set_value(const double value) {
    m_values = std::vector<double>{value};
  }

  /// \brief Sets the plaintext to store the given values
  /// \param[in] values Values to store in the plaintext
  inline void set_values(const std::vector<double>& values) {
    m_values = values;
  }

  /// \brief returns whether or not the plaintext stores a single value
  inline bool is_single_value() const { return num_values() == 1; }

  /// \brief Returns the number of values stored in the plaintext
  inline size_t num_values() const { return m_values.size(); }

  /// \brief Saves the cihertext to a protobuf ciphertext wrapper
  /// \param[out] proto_cipher Protobuf ciphertext wrapper to store the
  /// ciphertext
  inline void save(he_proto::Plaintext& proto_cipher) const {}

 private:
  std::vector<double> m_values;
};
}  // namespace he
}  // namespace ngraph
