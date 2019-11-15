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

namespace ngraph::runtime::aby {

class ABYExecutor {
 public:
  ABYExecutor(const std::string& role, const std::string& mpc_protocol,
              const std::string& hostname, std::size_t port,
              uint64_t security_level, uint32_t bit_length,
              uint32_t num_threads, const std::string& mg_algo_str,
              uint32_t reserve_num_gates);

  virtual ~ABYExecutor();

  inline ABYParty* get_aby_party() { return m_ABYParty; }

  inline BooleanCircuit* get_circuit() {
    m_sharings = m_ABYParty->GetSharings();
    auto circ = dynamic_cast<BooleanCircuit*>(
        m_sharings[m_aby_gc_protocol]->GetCircuitBuildRoutine());
    return circ;
  }

  void mask_input_unknown_relu_ciphers_batch(
      std::vector<he::HEType>& cipher_batch);

  void start_aby_circuit_unknown_relu_ciphers_batch(
      std::vector<he::HEType>& cipher_batch);

  /// If client, mean reduce zero centers the  data
  /// If server, makss with input mask values
  virtual void prepare_aby_circuit(const std::string& function,
                                   std::shared_ptr<he::HETensor>& tensor) {}

  /// If client, runs aby circuit
  /// If server, runs aby circuit and populates tensor with outputs
  virtual void run_aby_circuit(const std::string& function,
                               std::shared_ptr<he::HETensor>& tensor) = 0;

  void reset_party() { m_ABYParty->Reset(); }

 protected:
  size_t m_num_threads;

  std::vector<Sharing*> m_sharings{};

  e_role m_role;
  e_sharing m_aby_gc_protocol;
  e_mt_gen_alg m_mt_gen_alg;
  uint32_t m_aby_bitlen;
  uint64_t m_security_level;
  ABYParty* m_ABYParty;

  size_t m_lowest_coeff_modulus{0};
};

}  // namespace ngraph::runtime::aby