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

namespace ngraph::he {

/// \brief Unpacks complex values to real values
/// (a+bi, c+di) => (a,b,c,d)
/// \param[out] output Vector to store unpacked real values
/// \param[in] input Vector of complex values to unpack
void complex_vec_to_real_vec(std::vector<double>& output,
                             const std::vector<std::complex<double>>& input);

/// \brief Packs elements of input into complex values
/// (a,b,c,d) => (a+bi, c+di)
/// (a,b,c) => (a+bi, c+0i)
/// \param[out] output Vector to store packed complex values
/// \param[in] input Vector of real values to unpack
void real_vec_to_complex_vec(std::vector<std::complex<double>>& output,
                             const std::vector<double>& input);

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

bool param_originates_from_name(const op::Parameter& param,
                                const std::string& name);

pb::HETensor_ElementType type_to_pb_type(const element::Type& element_type);

element::Type pb_type_to_type(pb::HETensor_ElementType pb_type);

}  // namespace ngraph::he
