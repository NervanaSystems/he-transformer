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

#include "logging/ngraph_he_log.hpp"

#include <cstdint>
#include <string>

#include "ngraph/log.hpp"

namespace ngraph::runtime::he::logging {
// Parse log level (int64) from environment variable (char*)
int64_t log_level_str_to_int(const char* env_var_val) {
  if (env_var_val == nullptr) {
    return 0;
  }

  // Ideally we would use env_var / safe_strto64, but it is
  // hard to use here without pulling in a lot of dependencies,
  // so we use std:istringstream instead
  std::string min_log_level(env_var_val);
  std::istringstream ss(min_log_level);
  int64_t level;
  if (!(ss >> level)) {
    // Invalid vlog level setting, set level to default (0)
    level = 0;
  }

  return level;
}

int64_t min_ngraph_he_log_level() {
  const char* tf_env_var_val = std::getenv("NGRAPH_HE_LOG_LEVEL");
  return log_level_str_to_int(tf_env_var_val);
}
}  // namespace ngraph::runtime::he::logging
