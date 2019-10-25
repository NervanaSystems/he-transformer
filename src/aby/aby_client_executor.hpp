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

#include "aby/aby_executor.hpp"
#include "aby/kernel/relu_aby.hpp"
#include "he_tensor.hpp"
#include "he_type.hpp"
#include "logging/ngraph_he_log.hpp"
#include "ngraph/except.hpp"
#include "ngraph/util.hpp"
#include "seal/he_seal_client.hpp"

namespace ngraph {

namespace he {
class HESealClient;
}
namespace aby {

class ABYClientExecutor : public ABYExecutor {
 public:
  ABYClientExecutor(std::string mpc_protocol,
                    const he::HESealClient& he_seal_client,
                    std::string hostname = std::string("0.0.0.0"),
                    std::size_t port = 7766, uint64_t security_level = 128,
                    uint32_t bit_length = 64, uint32_t num_threads = 1,
                    std::string mg_algo_str = std::string("MT_OT"),
                    uint32_t reserve_num_gates = 65536,
                    const std::string& circuit_directory = "");

  ~ABYClientExecutor() = default;

  void prepare_aby_circuit(const std::string& function,
                           std::shared_ptr<he::HETensor>& tensor) override;

  void prepare_aby_relu_circuit(const std::string& function,
                                std::shared_ptr<he::HETensor>& tensor);

  void run_aby_circuit(const std::string& function,
                       std::shared_ptr<he::HETensor>& tensor) override;

  void run_aby_relu_circuit(const std::string& function,
                            std::shared_ptr<he::HETensor>& tensor);

 private:
  const he::HESealClient& m_he_seal_client;
};

}  // namespace aby
}  // namespace ngraph