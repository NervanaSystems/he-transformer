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

#include "he_util.hpp"

#include <complex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ngraph/check.hpp"
#include "ngraph/except.hpp"
#include "ngraph/util.hpp"
#include "protos/message.pb.h"

namespace ngraph::runtime::he {

void complex_vec_to_real_vec(std::vector<double>& output,
                             const std::vector<std::complex<double>>& input) {
  NGRAPH_CHECK(output.empty(), "Output vector is not empty");
  output.reserve(input.size() * 2);
  for (const std::complex<double>& value : input) {
    output.emplace_back(value.real());
    output.emplace_back(value.imag());
  }
}

void real_vec_to_complex_vec(std::vector<std::complex<double>>& output,
                             const std::vector<double>& input) {
  NGRAPH_CHECK(output.empty(), "Output vector is not empty");
  output.reserve(input.size() / 2);
  std::vector<double> complex_parts(2, 0);
  for (size_t i = 0; i < input.size(); ++i) {
    complex_parts[i % 2] = input[i];

    if (i % 2 == 1 || i == input.size() - 1) {
      output.emplace_back(
          std::complex<double>(complex_parts[0], complex_parts[1]));
      complex_parts = {0, 0};
    }
  }
}

bool flag_to_bool(const char* flag, bool default_value) {
  if (flag == nullptr) {
    return default_value;
  }
  static std::unordered_set<std::string> on_map{"1", "on", "y", "yes", "true"};
  static std::unordered_set<std::string> off_map{"0", "off", "n", "no",
                                                 "false"};
  std::string flag_str = to_lower(std::string(flag));

  if (on_map.find(flag_str) != on_map.end()) {
    return true;
  }
  if (off_map.find(flag_str) != off_map.end()) {
    return false;
  }
  throw ngraph_error("Unknown flag value " + std::string(flag));
}

int flag_to_int(const char* flag, int default_value) {
  if (flag == nullptr) {
    return default_value;
  }
  return std::stoi(std::string(flag));
}

double type_to_double(const void* src, const element::Type& element_type) {
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wswitch"
#pragma clang diagnostic error "-Wswitch-enum"
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
  }
#pragma clang diagnostic pop
}

bool param_originates_from_name(const op::Parameter& param,
                                const std::string& name) {
  if (param.get_name() == name) {
    return true;
  }
  return std::any_of(param.get_provenance_tags().begin(),
                     param.get_provenance_tags().end(),
                     [&](const std::string& tag) { return tag == name; });
}

pb::HETensor_ElementType type_to_pb_type(const element::Type& element_type) {
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wswitch"
#pragma clang diagnostic error "-Wswitch-enum"
  switch (element_type.get_type_enum()) {
    case element::Type_t::undefined: {
      return pb::HETensor::UNDEFINED;
    }
    case element::Type_t::dynamic: {
      return pb::HETensor::DYNAMIC;
    }
    case element::Type_t::boolean: {
      return pb::HETensor::BOOLEAN;
    }
    case element::Type_t::bf16: {
      return pb::HETensor::BF16;
    }
    case element::Type_t::f16: {
      return pb::HETensor::F16;
    }
    case element::Type_t::f32: {
      return pb::HETensor::F32;
    }
    case element::Type_t::f64: {
      return pb::HETensor::F64;
    }
    case element::Type_t::i8: {
      return pb::HETensor::I8;
    }
    case element::Type_t::i16: {
      return pb::HETensor::I16;
    }
    case element::Type_t::i32: {
      return pb::HETensor::I32;
    }
    case element::Type_t::i64: {
      return pb::HETensor::I64;
    }
    case element::Type_t::u8: {
      return pb::HETensor::U8;
    }
    case element::Type_t::u16: {
      return pb::HETensor::U16;
    }
    case element::Type_t::u32: {
      return pb::HETensor::U32;
    }
    case element::Type_t::u64: {
      return pb::HETensor::U64;
    }
  }
#pragma clang diagnostic pop
}

element::Type pb_type_to_type(pb::HETensor_ElementType pb_type) {
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wswitch"
#pragma clang diagnostic error "-Wswitch-enum"
  switch (pb_type) {
    case pb::HETensor::UNDEFINED:
    case pb::
        HETensor_ElementType_HETensor_ElementType_INT_MIN_SENTINEL_DO_NOT_USE_:
    case pb::
        HETensor_ElementType_HETensor_ElementType_INT_MAX_SENTINEL_DO_NOT_USE_: {
      return element::Type_t::undefined;
    }
    case pb::HETensor::DYNAMIC: {
      return element::Type_t::dynamic;
    }
    case pb::HETensor::BOOLEAN: {
      return element::Type_t::boolean;
    }
    case pb::HETensor::BF16: {
      return element::Type_t::bf16;
    }
    case pb::HETensor::F16: {
      return element::Type_t::f16;
    }
    case pb::HETensor::F32: {
      return element::Type_t::f32;
    }
    case pb::HETensor::F64: {
      return element::Type_t::f64;
    }
    case pb::HETensor::I8: {
      return element::Type_t::i8;
    }
    case pb::HETensor::I16: {
      return element::Type_t::i16;
    }
    case pb::HETensor::I32: {
      return element::Type_t::i32;
    }
    case pb::HETensor::I64: {
      return element::Type_t::i64;
    }
    case pb::HETensor::U8: {
      return element::Type_t::u8;
    }
    case pb::HETensor::U16: {
      return element::Type_t::u16;
    }
    case pb::HETensor::U32: {
      return element::Type_t::u32;
    }
    case pb::HETensor::U64: {
      return element::Type_t::u64;
    }
#pragma clang diagnostic pop
  }
}

}  // namespace ngraph::runtime::he
