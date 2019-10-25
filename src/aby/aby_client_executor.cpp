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

#include "aby/aby_client_executor.hpp"
#include "seal/kernel/subtract_seal.hpp"
#include "seal/seal_util.hpp"

namespace ngraph {
namespace aby {

ABYClientExecutor::ABYClientExecutor(
    std::string mpc_protocol, seal::CKKSEncoder& ckks_encoder,
    std::shared_ptr<seal::SEALContext> seal_context,
    const seal::Encryptor& encryptor, seal::Decryptor& decryptor,
    const ngraph::he::HESealEncryptionParameters& encryption_params,
    std::string hostname, std::size_t port, uint64_t security_level,
    uint32_t bit_length, uint32_t num_threads, std::string mg_algo_str,
    uint32_t reserve_num_gates, const std::string& circuit_directory)
    : ABYExecutor("client", mpc_protocol, hostname, port, security_level,
                  bit_length, num_threads, mg_algo_str, reserve_num_gates,
                  circuit_directory) {
  NGRAPH_HE_LOG(1) << "Started ABYClientExecutor";
}

void ABYClientExecutor::prepare_aby_circuit(
    const std::string& function, std::shared_ptr<he::HETensor>& tensor) {
  NGRAPH_HE_LOG(3) << "cleint prepare_aby_circuit";
}

void ABYClientExecutor::run_aby_circuit(const std::string& function,
                                        std::shared_ptr<he::HETensor>& tensor) {
  NGRAPH_HE_LOG(3) << "cleint prepare_aby_circuit";
}

}  // namespace aby
}  // namespace ngraph