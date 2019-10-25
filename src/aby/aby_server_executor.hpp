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
#include "seal/he_seal_executable.hpp"

namespace ngraph {

namespace he {
class HESealExecutable;
}
namespace aby {

class ABYServerExecutor : public ABYExecutor {
 public:
  ABYServerExecutor(he::HESealExecutable& he_seal_executable,
                    std::string mpc_protocol,
                    std::string hostname = std::string("localhost"),
                    std::size_t port = 7766, uint64_t security_level = 128,
                    uint32_t bit_length = 64, uint32_t num_threads = 2,
                    std::string mg_algo_str = std::string("MT_OT"),
                    uint32_t reserve_num_gates = 65536,
                    const std::string& circuit_directory = "");

  ~ABYServerExecutor() = default;

  std::shared_ptr<he::HETensor> generate_gc_mask(
      const ngraph::Shape& shape, bool plaintext_packing, bool complex_packing,
      const std::string& name, bool random = true, uint64_t default_value = 0);

  std::shared_ptr<he::HETensor> generate_gc_input_mask(
      const ngraph::Shape& shape, bool plaintext_packing, bool complex_packing,
      uint64_t default_value = 0);

  std::shared_ptr<he::HETensor> generate_gc_output_mask(
      const ngraph::Shape& shape, bool plaintext_packing, bool complex_packing,
      uint64_t default_value = 0);

  void prepare_aby_circuit(const std::string& function,
                           std::shared_ptr<he::HETensor>& tensor) override;

  void run_aby_circuit(const std::string& function,
                       std::shared_ptr<he::HETensor>& tensor) override;

  // TODO: remove
  void start_aby_circuit_unknown_relu_ciphers_batch(
      std::vector<he::HEType>& cipher_batch);

  // TODO: remove
  void mask_input_unknown_relu_ciphers_batch(
      std::vector<he::HEType>& cipher_batch);

 private:
  he::HESealExecutable& m_he_seal_executable;
  std::shared_ptr<he::HETensor> m_gc_input_mask;
  std::shared_ptr<he::HETensor> m_gc_output_mask;
};

}  // namespace aby
}  // namespace ngraph