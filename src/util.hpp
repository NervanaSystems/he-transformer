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
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ngraph/check.hpp"
#include "ngraph/except.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "node_wrapper.hpp"
#include "op/bounded_relu.hpp"
#include "protos/message.pb.h"

namespace ngraph {
namespace he {

inline std::string bool_to_string(const bool b) {
  std::ostringstream ss;
  ss << std::boolalpha << b;
  return ss.str();
}

/// \brief Interprets a string as a boolean value
/// \param[in] value String to interpret
/// \param[in] default_value Value to return if flag is not able to be parsed
/// \returns True if flag represents a True value, False otherwise
inline bool string_to_bool(const char* value, bool default_value = false) {
  if (value == nullptr) {
    return default_value;
  }
  static std::unordered_set<std::string> on_map{"1", "on", "y", "yes", "true"};
  static std::unordered_set<std::string> off_map{"0", "off", "n", "no",
                                                 "false"};
  std::string value_str = ngraph::to_lower(std::string(value));

  if (on_map.find(value_str) != on_map.end()) {
    return true;
  } else if (off_map.find(value_str) != off_map.end()) {
    return false;
  } else {
    throw ngraph_error("Unknown flag value " + std::string(value));
  }
}

inline bool string_to_bool(std::string value, bool default_value = false) {
  return string_to_bool(value.c_str(), default_value);
}

/// \brief Unpacks complex values to real values
/// (a+bi, c+di) => (a,b,c,d)
/// \param[out] output Vector to store unpacked real values
/// \param[in] input Vector of complex values to unpack
template <typename T>
inline void complex_vec_to_real_vec(std::vector<T>& output,
                                    const std::vector<std::complex<T>>& input) {
  NGRAPH_CHECK(output.empty(), "Output vector is not empty");
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
  NGRAPH_CHECK(output.empty(), "Output vector is not empty");
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

template <typename T>
inline std::unordered_map<std::string,
                          std::pair<std::string, std::vector<double>>>
map_to_double_map(
    const std::unordered_map<std::string,
                             std::pair<std::string, std::vector<T>>>& inputs) {
  std::unordered_map<std::string, std::pair<std::string, std::vector<double>>>
      outputs;

  for (const auto& [key, value] : inputs) {
    std::vector<double> double_inputs{value.second.begin(), value.second.end()};
    outputs.insert({key, std::make_pair(value.first, double_inputs)});
  }
  return outputs;
}

/// \brief Interprets a string as a boolean value
/// \param[in] flag Flag value
/// \param[in] default_value Value to return if flag is not able to be parsed
/// \returns True if flag represents a True value, False otherwise
bool flag_to_bool(const char* flag, bool default_value = false);

/// \brief Converts a type to a double using static_cast
/// Note, this means a reduction of range in int64 and uint64 values.
/// \param[in] src Source from which to read
/// \param[in] element_type Datatype to interpret source as
/// \returns double value
double type_to_double(const void* src, const element::Type& element_type);

bool param_originates_from_name(const ngraph::op::Parameter& param,
                                const std::string& name);

inline proto::Function node_to_proto_function(
    const NodeWrapper& node_wrapper,
    std::unordered_map<std::string, std::string> extra_configs = {}) {
  const Node& node = *node_wrapper.get_node();
  auto type_id = node_wrapper.get_typeid();

  nlohmann::json js = {{"function", node.description()}};
  if (type_id == OP_TYPEID::BoundedRelu) {
    const op::BoundedRelu* bounded_relu =
        static_cast<const op::BoundedRelu*>(&node);
    float alpha = bounded_relu->get_alpha();
    js["bound"] = alpha;
  }

  for (const auto& [key, value] : extra_configs) {
    js[key] = value;
  }

  proto::Function f;
  f.set_function(js.dump());
  return f;
}

}  // namespace he
}  // namespace ngraph
