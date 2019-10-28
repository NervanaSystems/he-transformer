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

#include <chrono>

#include "aby/aby_client_executor.hpp"
#include "nlohmann/json.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_util.hpp"

using json = nlohmann::json;

namespace ngraph {
namespace aby {

ABYClientExecutor::ABYClientExecutor(
    std::string mpc_protocol, const he::HESealClient& he_seal_client,
    std::string hostname, std::size_t port, uint64_t security_level,
    uint32_t bit_length, uint32_t num_threads, std::string mg_algo_str,
    uint32_t reserve_num_gates, const std::string& circuit_directory)
    : ABYExecutor("client", mpc_protocol, hostname, port, security_level,
                  bit_length, num_threads, mg_algo_str, reserve_num_gates,
                  circuit_directory),
      m_he_seal_client(he_seal_client) {
  m_lowest_coeff_modulus = m_he_seal_client.encryption_paramters()
                               .seal_encryption_parameters()
                               .coeff_modulus()[0]
                               .value();

  NGRAPH_HE_LOG(1) << "Started ABYClientExecutor";
}

void ABYClientExecutor::prepare_aby_circuit(
    const std::string& function, std::shared_ptr<he::HETensor>& tensor) {
  NGRAPH_HE_LOG(3) << "client prepare_aby_circuit with function " << function;
  json js = json::parse(function);
  auto name = js.at("function");

  if (name == "Relu") {
    prepare_aby_relu_circuit(function, tensor);
  } else {
    NGRAPH_ERR << "Unknown function name " << name;
    throw ngraph_error("Unknown function name");
  }
}

void ABYClientExecutor::run_aby_circuit(const std::string& function,
                                        std::shared_ptr<he::HETensor>& tensor) {
  NGRAPH_HE_LOG(3) << "client run_aby_circuit with function " << function;
  json js = json::parse(function);
  auto name = js.at("function");

  if (name == "Relu") {
    run_aby_relu_circuit(function, tensor);
  } else {
    NGRAPH_ERR << "Unknown function name " << name;
    throw ngraph_error("Unknown function name");
  }
}

void ABYClientExecutor::prepare_aby_relu_circuit(
    const std::string& function, std::shared_ptr<he::HETensor>& tensor) {
  NGRAPH_HE_LOG(3) << "prepare_aby_relu_circuit";
}

void ABYClientExecutor::run_aby_relu_circuit(
    const std::string& function, std::shared_ptr<he::HETensor>& tensor) {
  NGRAPH_HE_LOG(3) << "run_aby_relu_circuit";

  auto& tensor_data = tensor->data();
  size_t batch_size = tensor_data[0].batch_size();
  uint64_t tensor_size = static_cast<uint64_t>(tensor_data.size() * batch_size);

  std::vector<double> relu_vals(tensor_size);
  size_t num_bytes = tensor_size * tensor->get_element_type().size();
  tensor->read(relu_vals.data(), num_bytes);

  std::vector<uint64_t> client_gc_vals(tensor_size);
  for (size_t i = 0; i < tensor_size; ++i) {
    // TOOD: check
    he::HEType& he_type = tensor_data[i % tensor_data.size()];
    NGRAPH_CHECK(he_type.is_ciphertext(), "HEType is not ciphertext");
    auto scale = he_type.get_ciphertext()->scale();

    // Reduce values to range (-q/(2*scale), q/(2*scale))
    auto relu_val = aby::mod_reduce_zero_centered(
        relu_vals[i], m_lowest_coeff_modulus / scale);

    // Turn SEAL's mapping (-q/(2*scale), q/(2*scale)) to (0,q)
    uint64_t relu_int_val;
    if (relu_val <= 0) {
      relu_int_val = std::round(relu_val * scale + m_lowest_coeff_modulus);
    } else {
      relu_int_val = std::round(relu_val * scale);
    }

    client_gc_vals[i] = relu_int_val;
  }

  NGRAPH_HE_LOG(3) << "Client creating relu circuit";
  std::vector<uint64_t> zeros(tensor_size, 0);

  BooleanCircuit* circ = get_circuit();
  auto* relu_out =
      ngraph::aby::relu_aby(*circ, tensor_size, zeros, client_gc_vals, zeros,
                            m_aby_bitlen, m_lowest_coeff_modulus);
  NGRAPH_HE_LOG(3) << "Client executing relu circuit";

  auto t1 = std::chrono::high_resolution_clock::now();
  m_ABYParty->ExecCircuit();
  auto t2 = std::chrono::high_resolution_clock::now();
  NGRAPH_HE_LOG(3)
      << "Client executing circuit took "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << "us";

  uint32_t out_bitlen_relu, result_count;
  uint64_t* out_vals_relu;  // output of circuit this value will be encrypted
                            // and sent to server
  relu_out->get_clear_value_vec(&out_vals_relu, &out_bitlen_relu,
                                &result_count);
  NGRAPH_INFO << "result_count " << result_count;
  for (size_t i = 0; i < result_count; ++i) {
    NGRAPH_INFO << out_vals_relu[i];
  }

  double scale = m_he_seal_client.scale();

  NGRAPH_CHECK(result_count == tensor->data().size(),
               "Wrong number of ABY result values");

  NGRAPH_INFO << "tensor size " << tensor->data().size();

  for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
    he::HEPlaintext post_relu_vals(batch_size);
    for (size_t fill_idx = 0; fill_idx < batch_size; ++fill_idx) {
      size_t out_idx = result_idx + fill_idx * result_count;
      uint64_t out_val = out_vals_relu[out_idx];
      double d_out_val =
          ngraph::aby::uint64_to_double(out_val, m_lowest_coeff_modulus, scale);
      post_relu_vals[fill_idx] = d_out_val;
    }

    auto cipher = he::HESealBackend::create_empty_ciphertext();
    NGRAPH_INFO << "Encrypting " << post_relu_vals << " at scale " << scale;

    ngraph::he::encrypt(
        cipher, post_relu_vals,
        m_he_seal_client.get_context()->first_parms_id(), ngraph::element::f64,
        scale, *m_he_seal_client.get_ckks_encoder(),
        *m_he_seal_client.get_encryptor(), m_he_seal_client.complex_packing());

    tensor->data(result_idx).set_ciphertext(cipher);
  }

  reset_party();
}
}  // namespace aby
}  // namespace ngraph