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
#include "he_tensor.hpp"
#include "he_type.hpp"
#include "logging/ngraph_he_log.hpp"
#include "ngraph/except.hpp"
#include "ngraph/util.hpp"
#include "seal/he_seal_executable.hpp"

namespace ngraph::runtime::he {
class HESealExecutable;
}
namespace ngraph::runtime::aby {

class ABYServerExecutor : public ABYExecutor {
 public:
  ABYServerExecutor(he::HESealExecutable& he_seal_executable,
                    const std::string& mpc_protocol,
                    std::string hostname = std::string("localhost"),
                    std::size_t port = 34001, uint64_t security_level = 128,
                    uint32_t bit_length = 64, uint32_t num_threads = 1,
                    std::string mg_algo_str = std::string("MT_OT"),
                    uint32_t reserve_num_gates = 65536);

  ~ABYServerExecutor() = default;

  std::shared_ptr<he::HETensor> generate_gc_mask(
      const Shape& shape, bool plaintext_packing, bool complex_packing,
      const std::string& name, bool random = true, uint64_t default_value = 0);

  std::shared_ptr<he::HETensor> generate_gc_input_mask(
      const Shape& shape, bool plaintext_packing, bool complex_packing,
      uint64_t default_value = 0);

  std::shared_ptr<he::HETensor> generate_gc_output_mask(
      const Shape& shape, bool plaintext_packing, bool complex_packing,
      uint64_t default_value = 0);

  void prepare_aby_circuit(const std::string& function,
                           std::shared_ptr<he::HETensor>& tensor) override;

  void run_aby_circuit(const std::string& function,
                       std::shared_ptr<he::HETensor>& tensor) override;

  void post_process_aby_circuit(const std::string& function,
                                std::shared_ptr<he::HETensor>& tensor);

  // Relu functions
  void prepare_aby_relu_circuit(std::vector<he::HEType>& cipher_batch);
  void run_aby_relu_circuit(std::vector<he::HEType>& cipher_batch);
  void post_process_aby_relu_circuit(std::shared_ptr<he::HETensor>& tensor);

  // Bounded Relu functions
  void prepare_aby_bounded_relu_circuit(std::vector<he::HEType>& cipher_batch,
                                        double bound);
  void run_aby_bounded_relu_circuit(std::vector<he::HEType>& cipher_batch,
                                    double bound);
  void post_process_aby_bounded_relu_circuit(
      std::shared_ptr<he::HETensor>& tensor, double bound);

 private:
  he::HESealExecutable& m_he_seal_executable;
  std::shared_ptr<he::HETensor> m_gc_input_mask;
  std::shared_ptr<he::HETensor> m_gc_output_mask;

  std::default_random_engine m_random_generator;
  int64_t m_rand_max;
  std::uniform_int_distribution<int64_t> m_random_distribution;
};

}  // namespace ngraph::runtime::aby