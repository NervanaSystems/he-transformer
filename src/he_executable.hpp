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

#include <memory>
#include <vector>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/util.hpp"

namespace ngraph {
namespace runtime {
namespace he {

class HEExecutable : public Executable {
 public:
  HEExecutable(const std::shared_ptr<Function>& function,
               bool enable_performance_collection = false);

  bool call(const std::vector<std::shared_ptr<Tensor>>& outputs,
            const std::vector<std::shared_ptr<Tensor>>& intputs) override;

  std::vector<PerformanceCounter> get_performance_data() const override;

 private:
  std::unordered_map<const Node*, stopwatch> m_timer_map;
};

}  // namespace he
}  // namespace runtime
}  // namespace ngraph
