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

#include <complex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ngraph/check.hpp"
#include "ngraph/except.hpp"
#include "ngraph/util.hpp"
#include "protos/message.pb.h"

namespace ngraph {
namespace he {

/// \brief Unpacks complex values to real values
/// (a+bi, c+di) => (a,b,c,d)
/// \param[out] output Vector to store unpacked real values
/// \param[in] input Vector of complex values to unpack
template <typename T>
inline void complex_vec_to_real_vec(std::vector<T>& output,
                                    const std::vector<std::complex<T>>& input) {
  NGRAPH_CHECK(output.size() == 0, "Output vector is not empty");
  output.reserve(input.size() * 2);
  for (const std::complex<T>& value : input) {
    output.emplace_back(value.real());
    output.emplace_back(value.imag());
  }
}

/// \brief Packs elements of input into complex values
/// (a,b,c,d) => (a+bi, c+di)
/// (a,b,c) => (a+bi, c+0i)
/// \param[out] output Vector to store packed complex values
/// \param[in] input Vector of real values to unpack
template <typename T>
inline void real_vec_to_complex_vec(std::vector<std::complex<T>>& output,
                                    const std::vector<T>& input) {
  NGRAPH_CHECK(output.size() == 0, "Output vector is not empty");
  output.reserve(input.size() / 2);
  std::vector<T> complex_parts(2, 0);
  for (size_t i = 0; i < input.size(); ++i) {
    complex_parts[i % 2] = input[i];

    if (i % 2 == 1 || i == input.size() - 1) {
      output.emplace_back(std::complex<T>(complex_parts[0], complex_parts[1]));
      complex_parts = {T(0), T(0)};
    }
  }
}

/// \brief Interprets a string as a boolean value
/// \param[in] flag Flag value
/// \param[in] default_value Value to return if flag is not able to be parsed
/// \returns True if flag represents a True value, False otherwise
inline bool flag_to_bool(const char* flag, bool default_value = false) {
  if (flag == nullptr) {
    return default_value;
  }
  static std::unordered_set<std::string> on_map{"1", "on", "y", "yes", "true"};
  static std::unordered_set<std::string> off_map{"0", "off", "n", "no",
                                                 "false"};
  std::string flag_str = ngraph::to_lower(std::string(flag));

  if (on_map.find(flag_str) != on_map.end()) {
    return true;
  } else if (off_map.find(flag_str) != off_map.end()) {
    return false;
  } else {
    throw ngraph_error("Unknown flag value " + std::string(flag));
  }
}

/// \brief Converts a type to a double using static_cast
/// Note, this means a reduction of range in int64 and uint64 values.
/// \param[in] src Source from which to read
/// \param[in] element_type Datatype to interpret source as
/// \returns double value
inline double type_to_double(const void* src,
                             const element::Type& element_type) {
  switch (element_type.get_type_enum()) {
    case element::Type_t::f32:
      return static_cast<double>(*static_cast<const float*>(src));
      break;
    case element::Type_t::f64:
      return static_cast<double>(*static_cast<const double*>(src));
      break;
    case element::Type_t::i32:
      return static_cast<double>(*static_cast<const int32_t*>(src));
      break;
    case element::Type_t::i64:
      return static_cast<double>(*static_cast<const int64_t*>(src));
      break;
    case element::Type_t::i8:
    case element::Type_t::i16:
    case element::Type_t::u8:
    case element::Type_t::u16:
    case element::Type_t::u32:
    case element::Type_t::u64:
    case element::Type_t::dynamic:
    case element::Type_t::undefined:
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::boolean:
      NGRAPH_CHECK(false, "Unsupported element type ", element_type);
      break;
  }
  NGRAPH_CHECK(false, "Unsupported element type ", element_type);
  return 0.0;
}

/// \brief Converts a vector of values of given type to a vector of doubles
/// \param[in] src Pointer to beginning of input vector
/// \param[in] element_type Datatype of input vector
/// \param[in] n Number of elements in the input vector
/// \returns Vector of double values
inline std::vector<double> type_vec_to_double_vec(
    const void* src, const element::Type& element_type, size_t n) {
  std::vector<double> ret(n);
  char* src_with_offset = static_cast<char*>(const_cast<void*>(src));
  for (size_t i = 0; i < n; ++i) {
    ret[i] = type_to_double(src_with_offset, element_type);
    ++src_with_offset;
  }
  return ret;
}

/// \brief Writes a vector of double values interpreted as a different datatype
/// to a target
/// \param[in] target Pointer to write to
/// \param[in] element_type Datatype of interpret input as
/// \param[in] input Vector of input values
inline void double_vec_to_type_vec(void* target,
                                   const element::Type& element_type,
                                   const std::vector<double>& input) {
  NGRAPH_CHECK(input.size() > 0, "Input has no values");
  size_t count = input.size();
  size_t type_byte_size = element_type.size();

  switch (element_type.get_type_enum()) {
    case element::Type_t::f32: {
      std::vector<float> float_values{input.begin(), input.end()};
      const void* type_values_src =
          static_cast<const void*>(float_values.data());
      std::memcpy(target, type_values_src, type_byte_size * count);
      break;
    }
    case element::Type_t::f64: {
      const void* type_values_src =
          static_cast<const void*>(input.data());
      std::memcpy(target, type_values_src, type_byte_size * count);
      break;
    }
    case element::Type_t::i64: {
      std::vector<int64_t> int64_values(input.size());
      for (size_t i = 0; i < input.size(); ++i) {
        int64_values[i] = std::round(input[i]);
      }
      const void* type_values_src =
          static_cast<const void*>(int64_values.data());
      std::memcpy(target, type_values_src, type_byte_size * count);
      break;
    }
    case element::Type_t::i8:
    case element::Type_t::i16:
    case element::Type_t::i32:
    case element::Type_t::u8:
    case element::Type_t::u16:
    case element::Type_t::u32:
    case element::Type_t::u64:
    case element::Type_t::dynamic:
    case element::Type_t::undefined:
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::boolean:
      NGRAPH_CHECK(false, "Unsupported element type ", element_type);
      break;
  }
}

template <typename T>
inline std::unordered_map<std::string,
                          std::pair<std::string, std::vector<double>>>
map_to_double_map(
    const std::unordered_map<std::string,
                             std::pair<std::string, std::vector<T>>>& inputs) {
  std::unordered_map<std::string, std::pair<std::string, std::vector<double>>>
      outputs;

  for (const auto& elem : inputs) {
    std::vector<double> double_inputs{elem.second.second.begin(),
                                      elem.second.second.end()};
    outputs.insert(
        {elem.first, std::make_pair(elem.second.first, double_inputs)});
  }
  return outputs;
}

inline ngraph::Shape proto_shape_to_ngraph_shape(
    const google::protobuf::RepeatedField<google::protobuf::uint64>&
        proto_shape) {
  std::vector<uint64_t> dims{proto_shape.begin(), proto_shape.end()};
  return ngraph::Shape{dims};
}

inline bool param_originates_from_name(const ngraph::op::Parameter& param,
                                       const std::string& name) {
  if (param.get_name() == name) {
    return true;
  }
  return std::any_of(param.get_provenance_tags().begin(),
                     param.get_provenance_tags().end(),
                     [&](const std::string& tag) { return tag == name; });
}

}  // namespace he
}  // namespace ngraph