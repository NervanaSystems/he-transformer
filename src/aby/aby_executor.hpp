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

#include "aby/kernel/relu_aby.hpp"
#include "abycore/aby/abyparty.h"
#include "abycore/circuit/booleancircuits.h"
#include "abycore/circuit/share.h"
#include "abycore/sharing/sharing.h"
#include "he_tensor.hpp"
#include "he_type.hpp"
#include "logging/ngraph_he_log.hpp"
#include "ngraph/except.hpp"
#include "ngraph/util.hpp"

namespace ngraph {
namespace aby {

class ABYExecutor {
 public:
  ABYExecutor(std::string role, std::string mpc_protocol,
              std::string hostname = std::string("localhost"),
              std::size_t port = 7766, uint64_t security_level = 128,
              uint32_t bit_length = 64, uint32_t num_threads = 2,
              std::string mg_algo_str = std::string("MT_OT"),
              uint32_t reserve_num_gates = 65536,
              const std::string& circuit_directory = "");

  virtual ~ABYExecutor();

  inline ABYParty* get_aby_party() { return m_ABYParty; }

  inline BooleanCircuit* get_circuit() {
    std::vector<Sharing*>& sharings = m_ABYParty->GetSharings();
    auto circ = dynamic_cast<BooleanCircuit*>(
        sharings[m_aby_gc_protocol]->GetCircuitBuildRoutine());
    return circ;
  }

  void mask_input_unknown_relu_ciphers_batch(
      std::vector<he::HEType>& cipher_batch);

  void start_aby_circuit_unknown_relu_ciphers_batch(
      std::vector<he::HEType>& cipher_batch);

  /// If client, mean reduce zero centers the  data
  /// If server, makss with input mask values
  virtual void prepare_aby_circuit(const std::string& function,
                                   std::shared_ptr<he::HETensor>& tensor) = 0;

  /// If client, runs aby circuit
  /// If server, runs aby circuit and populates tensor with outputs
  virtual void run_aby_circuit(const std::string& function,
                               std::shared_ptr<he::HETensor>& tensor) = 0;

 protected:
  size_t m_num_threads;
  bool m_mask_gc_inputs;
  bool m_mask_gc_outputs;

  e_role m_role;
  e_sharing m_aby_gc_protocol;
  e_mt_gen_alg m_mt_gen_alg;
  uint32_t m_aby_bitlen;
  uint64_t m_security_level;
  ABYParty* m_ABYParty;

  size_t m_lowest_coeff_modulus;
};

}  // namespace aby
}  // namespace ngraph