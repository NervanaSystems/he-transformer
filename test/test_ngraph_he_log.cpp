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

#include "gtest/gtest.h"
#include "logging/ngraph_he_log.hpp"

namespace ngraph::runtime::he {

TEST(ngraph_he_log, log_level_str_to_int) {
  EXPECT_EQ(uint64_t{0}, logging::log_level_str_to_int(nullptr));

  std::string invalid_str{"DUMMY"};
  EXPECT_EQ(uint64_t{0}, logging::log_level_str_to_int(invalid_str.c_str()));
}

}  // namespace ngraph::runtime::he
