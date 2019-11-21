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

#include "aby/aby_executor.hpp"

#include "seal/kernel/subtract_seal.hpp"
#include "seal/seal_util.hpp"

namespace ngraph::runtime::aby {

ABYExecutor::ABYExecutor(const std::string& role,
                         const std::string& mpc_protocol,
                         const std::string& hostname, std::size_t port,
                         uint64_t security_level, uint32_t bit_length,
                         uint32_t num_threads, uint32_t num_parties,
                         const std::string& mg_algo_str,
                         uint32_t reserve_num_gates)
    : m_num_threads{num_threads},
      m_num_parties{num_parties},
      m_aby_bitlen{bit_length} {
  static std::map<std::string, e_role> role_map{{"server", SERVER},
                                                {"client", CLIENT}};

  auto role_it = role_map.find(ngraph::to_lower(role));
  NGRAPH_CHECK(role_it != role_map.end(), "Unknown role ", role);
  m_role = role_it->second;

  static std::map<std::string, e_sharing> protocol_map{{"yao", S_YAO},
                                                       {"gmw", S_BOOL}};

  auto protocol_it = protocol_map.find(ngraph::to_lower(mpc_protocol));
  NGRAPH_CHECK(role_it != role_map.end(), "Unknown role ", role);
  NGRAPH_CHECK(protocol_it != protocol_map.end(), "Unknown mpc_protocol ",
               mpc_protocol);
  m_aby_gc_protocol = protocol_it->second;

  NGRAPH_CHECK(mg_algo_str == "MT_OT", "Unknown mg_algo_str ", mg_algo_str);
  m_mt_gen_alg = MT_OT;

  NGRAPH_CHECK(security_level == 128, "Unsupported security level ",
               security_level);
  m_security_level = security_level;

  NGRAPH_INFO << "Creating " << num_parties << " ABYParties with role " << role
              << " at " << hostname << ":" << port;

  m_ABYParties.resize(num_parties);
  for (size_t idx = 0; idx < num_parties; ++idx) {
    m_ABYParties[idx] = new ABYParty(
        m_role, hostname, port + idx, get_sec_lvl(m_security_level), bit_length,
        m_num_threads, m_mt_gen_alg, reserve_num_gates);
  }
  m_sharings.resize(num_parties);

  // TODO(fboemer): connect and base OTs?
  //  m_ABYParty_server->ConnectAndBaseOTs();
  NGRAPH_HE_LOG(1) << "Started ABYParty with role " << role << ", "
                   << num_parties << " parties";
}

// TODO(fboemer): delete ABYParty
ABYExecutor::~ABYExecutor() = default;

}  // namespace ngraph::runtime::aby