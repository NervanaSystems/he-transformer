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
using HEPlaintext = std::vector<double>;

inline void save(const HEPlaintext& plaintext, he_proto::Plaintext& proto) {
  for (const auto& value : plaintext) {
    proto.add_value(value);
  }
}

inline std::ostream& operator<<(std::ostream& os, const HEPlaintext& plain) {
  os << "HEPlaintext(";
  for (const auto& value : plain) {
    os << value << " ";
  }
  os << ")";
  return os;
}
}  // namespace he
}  // namespace ngraph
