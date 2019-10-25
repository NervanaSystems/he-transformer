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

#include "aby/aby_server_executor.hpp"
#include "nlohmann/json.hpp"
#include "seal/kernel/subtract_seal.hpp"
#include "seal/seal_util.hpp"

using json = nlohmann::json;

namespace ngraph {
namespace aby {

ABYServerExecutor::ABYServerExecutor(
    he::HESealExecutable& he_seal_executable, std::string mpc_protocol,
    std::string hostname, std::size_t port, uint64_t security_level,
    uint32_t bit_length, uint32_t num_threads, std::string mg_algo_str,
    uint32_t reserve_num_gates, const std::string& circuit_directory)
    : ABYExecutor("server", mpc_protocol, hostname, port, security_level,
                  bit_length, num_threads, mg_algo_str, reserve_num_gates,
                  circuit_directory),
      m_he_seal_executable{he_seal_executable} {
  m_lowest_coeff_modulus = m_he_seal_executable.he_seal_backend()
                               .get_encryption_parameters()
                               .seal_encryption_parameters()
                               .coeff_modulus()[0]
                               .value();
  m_rand_max = static_cast<int64_t>(m_lowest_coeff_modulus - 1);
  m_random_distribution = std::uniform_int_distribution<int64_t>{0, m_rand_max};
}

std::shared_ptr<he::HETensor> ABYServerExecutor::generate_gc_mask(
    const ngraph::Shape& shape, bool plaintext_packing, bool complex_packing,
    const std::string& name, bool random, uint64_t default_value) {
  auto tensor = std::make_shared<he::HETensor>(
      element::i64, shape, plaintext_packing, complex_packing, false,
      m_he_seal_executable.he_seal_backend(), name);

  std::vector<uint64_t> rand_vals(tensor->get_element_count());
  if (random) {
    auto random_gen = [this]() {
      return m_random_distribution(m_random_generator);
    };
    std::generate(rand_vals.begin(), rand_vals.end(), random_gen);
  } else {
    rand_vals = std::vector<uint64_t>(rand_vals.size(), default_value);
  }
  tensor->write(rand_vals.data(), rand_vals.size() * sizeof(uint64_t));

  return tensor;
}

std::shared_ptr<he::HETensor> ABYServerExecutor::generate_gc_input_mask(
    const ngraph::Shape& shape, bool plaintext_packing, bool complex_packing,
    uint64_t default_value) {
  return generate_gc_mask(
      shape, plaintext_packing, complex_packing, "gc_input_mask",
      m_he_seal_executable.he_seal_backend().mask_gc_inputs(), default_value);
}

std::shared_ptr<he::HETensor> ABYServerExecutor::generate_gc_output_mask(
    const ngraph::Shape& shape, bool plaintext_packing, bool complex_packing,
    uint64_t default_value) {
  return generate_gc_mask(
      shape, plaintext_packing, complex_packing, "gc_output_mask",
      m_he_seal_executable.he_seal_backend().mask_gc_outputs(), default_value);
}

void ABYServerExecutor::mask_input_unknown_relu_ciphers_batch(
    std::vector<he::HEType>& cipher_batch) {
  NGRAPH_HE_LOG(3) << "mask_input_unknown_relu_ciphers_batch ";

  bool plaintext_packing = cipher_batch[0].plaintext_packing();
  bool complex_packing = cipher_batch[0].complex_packing();
  size_t batch_size = cipher_batch[0].batch_size();

  NGRAPH_INFO << "Generating gc input mask";

  m_gc_input_mask =
      generate_gc_input_mask(Shape{batch_size, cipher_batch.size()},
                             plaintext_packing, complex_packing);

  NGRAPH_INFO << "Generating gc output mask";

  m_gc_output_mask =
      generate_gc_output_mask(Shape{batch_size, cipher_batch.size()},
                              plaintext_packing, complex_packing);

  std::vector<double> scales(cipher_batch.size());

  for (size_t i = 0; i < cipher_batch.size(); ++i) {
    auto& he_type = cipher_batch[i];
    auto& gc_input_mask = m_gc_input_mask->data(i);
    NGRAPH_CHECK(he_type.is_ciphertext(), "HEType is not ciphertext");

    auto cipher = he_type.get_ciphertext();

    NGRAPH_INFO << "Mod switching to lowest";
    // Swith modulus to lowest values since mask values are drawn
    // from (-q/2, q/2) for q the lowest coeff modulus
    m_he_seal_executable.he_seal_backend().rescale_to_lowest(*cipher);

    // Divide by scale so we can encode at the same scale as existing
    // ciphertext
    double scale = cipher->ciphertext().scale();
    scales[i] = scale;

    NGRAPH_INFO << "scaling input mask";

    he::HEPlaintext scaled_gc_input_mask(gc_input_mask.get_plaintext());
    for (size_t mask_idx = 0; mask_idx < scaled_gc_input_mask.size();
         ++mask_idx) {
      scaled_gc_input_mask[mask_idx] /= scale;
    }

    NGRAPH_INFO << "scalar_subtract_seal";
    scalar_subtract_seal(*cipher, scaled_gc_input_mask, cipher,
                         he_type.complex_packing(),
                         m_he_seal_executable.he_seal_backend());
  }

  for (const auto& scale : scales) {
    NGRAPH_CHECK(std::abs(scale - scales[0]) < 1e-3f, "Scale ", scale,
                 " does not match first scale ", scales[0]);
  }
}

void ABYServerExecutor::start_aby_circuit_unknown_relu_ciphers_batch(
    std::vector<he::HEType>& cipher_batch) {
  NGRAPH_INFO << "start_aby_circuit_unknown_relu_ciphers_batch ";

  // TODO: bounded relu?

  uint32_t num_aby_vals = cipher_batch.size() * cipher_batch[0].batch_size();
  std::vector<uint64_t> zeros(num_aby_vals, 0);
  std::vector<uint64_t> gc_input_mask_vals(num_aby_vals);
  std::vector<uint64_t> gc_output_mask_vals(num_aby_vals);
  std::vector<uint64_t> bound_vals(num_aby_vals);

  m_gc_input_mask->read(gc_input_mask_vals.data(),
                        num_aby_vals * sizeof(uint64_t));
  m_gc_output_mask->read(gc_output_mask_vals.data(),
                         num_aby_vals * sizeof(uint64_t));

  NGRAPH_HE_LOG(3) << "Server creating relu circuit";
  auto circ = get_circuit();
  ngraph::aby::relu_aby(circ, num_aby_vals, gc_input_mask_vals, zeros,
                        gc_output_mask_vals, m_aby_bitlen,
                        m_lowest_coeff_modulus);

  NGRAPH_HE_LOG(3) << "server executing relu circuit";
  m_ABYParty->ExecCircuit();
}

void ABYServerExecutor::prepare_aby_circuit(
    const std::string& function, std::shared_ptr<he::HETensor>& tensor) {
  NGRAPH_HE_LOG(3) << "server prepare_aby_circuit with funciton " << function;
  json js = json::parse(function);
  auto name = js.at("function");

  if (name == "Relu") {
    mask_input_unknown_relu_ciphers_batch(tensor->data());
  } else {
    NGRAPH_ERR << "Unknown function name " << name;
    throw ngraph_error("Unknown function name");
  }
}

void ABYServerExecutor::run_aby_circuit(const std::string& function,
                                        std::shared_ptr<he::HETensor>& tensor) {
  NGRAPH_HE_LOG(3) << "server run_aby_circuit with funciton " << function;

  json js = json::parse(function);
  auto name = js.at("function");
  if (name == "Relu") {
    start_aby_circuit_unknown_relu_ciphers_batch(tensor->data());
  } else {
    NGRAPH_ERR << "Unknown function name " << name;
    throw ngraph_error("Unknown function name");
  }
}

}  // namespace aby
}  // namespace ngraph