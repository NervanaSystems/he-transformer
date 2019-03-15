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

#include "he_backend.hpp"
#include "he_tensor.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/util.hpp"
#include "node_wrapper.hpp"

namespace ngraph {
namespace runtime {
namespace he {

class HEExecutable : public Executable {
 public:
  HEExecutable(const std::shared_ptr<Function>& function,
               bool enable_performance_collection,
               const runtime::he::HEBackend* he_backend, bool encrypt_data,
               bool encrypt_model, bool batch_data);

  bool call(const std::vector<std::shared_ptr<Tensor>>& outputs,
            const std::vector<std::shared_ptr<Tensor>>& inputs) override;

  void he_validate(
      const std::vector<std::shared_ptr<runtime::he::HETensor>>& outputs,
      const std::vector<std::shared_ptr<runtime::he::HETensor>>& inputs);

  std::vector<PerformanceCounter> get_performance_data() const override;

 private:
  bool m_encrypt_data;
  bool m_batch_data;
  bool m_encrypt_model;
  bool m_is_compiled = false;
  const HEBackend* m_he_backend = nullptr;  // TODO: replace with context
  std::unordered_map<const Node*, stopwatch> m_timer_map;
  std::vector<NodeWrapper> m_wrapped_nodes;

  void generate_calls(const element::Type& type, const NodeWrapper& op,
                      const std::vector<std::shared_ptr<HETensor>>& outputs,
                      const std::vector<std::shared_ptr<HETensor>>& inputs);
};

}  // namespace he
}  // namespace runtime
}  // namespace ngraph
