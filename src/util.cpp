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

#include "util.hpp"

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

bool flag_to_bool(const char* flag, bool default_value) {
  if (flag == nullptr) {
    return default_value;
  }
  static std::unordered_set<std::string> on_map{"1", "on", "y", "yes", "true"};
  static std::unordered_set<std::string> off_map{"0", "off", "n", "no",
                                                 "false"};
  std::string flag_str = ngraph::to_lower(std::string(flag));

  if (on_map.find(flag_str) != on_map.end()) {
    return true;
  }
  if (off_map.find(flag_str) != off_map.end()) {
    return false;
  }
  throw ngraph_error("Unknown flag value " + std::string(flag));
}

double type_to_double(const void* src, const element::Type& element_type) {
  switch (element_type.get_type_enum()) {
    case element::Type_t::f32: {
      return static_cast<double>(*static_cast<const float*>(src));
    }
    case element::Type_t::f64: {
      return static_cast<double>(*static_cast<const double*>(src));
    }
    case element::Type_t::i32: {
      return static_cast<double>(*static_cast<const int32_t*>(src));
    }
    case element::Type_t::i64: {
      return static_cast<double>(*static_cast<const int64_t*>(src));
    }
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

bool param_originates_from_name(const ngraph::op::Parameter& param,
                                const std::string& name) {
  if (param.get_name() == name) {
    return true;
  }
  return std::any_of(param.get_provenance_tags().begin(),
                     param.get_provenance_tags().end(),
                     [&](const std::string& tag) { return tag == name; });
}

}  // namespace ngraph::he
