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

#include <cstdint>
#include <string>

#include "ngraph/log.hpp"

namespace ngraph::runtime::he::logging {
// Parse log level (int64) from environment variable (char*)
int64_t log_level_str_to_int(const char* env_var_val);

int64_t min_ngraph_he_log_level();

}  // namespace ngraph::runtime::he::logging

#define NGRAPH_HE_VLOG_IS_ON(lvl) \
  ((lvl) <= ngraph::runtime::he::logging::min_ngraph_he_log_level())

#define NGRAPH_HE_LOG(lvl) \
  if (NGRAPH_HE_VLOG_IS_ON(lvl)) NGRAPH_INFO
// Comment to avoid backslash-newline at end of file warning
