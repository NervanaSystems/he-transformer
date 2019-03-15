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

#include <memory>
#include <vector>

#include "he_executable.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

runtime::he::HEExecutable::HEExecutable(const shared_ptr<Function>& function,
                                        bool enable_performance_collection) {}

bool runtime::he::HEExecutable::call(
    const vector<shared_ptr<runtime::Tensor>>& outputs,
    const vector<shared_ptr<runtime::Tensor>>& inputs) {
  return false;
}

vector<runtime::PerformanceCounter>
runtime::he::HEExecutable::get_performance_data() const {
  vector<runtime::PerformanceCounter> rc;
  for (const pair<const Node*, stopwatch> p : m_timer_map) {
    rc.emplace_back(p.first->get_name().c_str(),
                    p.second.get_total_microseconds(),
                    p.second.get_call_count());
  }
  return rc;
}
