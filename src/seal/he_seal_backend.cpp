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

#include "seal/he_seal_backend.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>

#include "logging/ngraph_he_log.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "seal/he_seal_executable.hpp"
#include "seal/seal.h"
#include "seal/seal_util.hpp"

using json = nlohmann::json;

extern "C" void ngraph_register_he_seal_backend() {
  ngraph::runtime::BackendManager::register_backend(
      "HE_SEAL", [](const std::string& /* config */) {
        return std::make_shared<ngraph::runtime::he::HESealBackend>();
      });
}

namespace ngraph::runtime::he {

HESealBackend::HESealBackend()
    : HESealBackend(HESealEncryptionParameters::parse_config_or_use_default(
          getenv("NGRAPH_HE_SEAL_CONFIG"))) {}

HESealBackend::HESealBackend(HESealEncryptionParameters parms)
    : m_encryption_params(std::move(parms)) {
  generate_context();
}

void HESealBackend::generate_context() {
  seal::sec_level_type sec_level =
      seal_security_level(m_encryption_params.security_level());

  m_context = seal::SEALContext::Create(
      m_encryption_params.seal_encryption_parameters(), true, sec_level);

  auto context_data = m_context->key_context_data();

  m_keygen = std::make_shared<seal::KeyGenerator>(m_context);
  if (m_context->using_keyswitching()) {
    m_relin_keys = std::make_shared<seal::RelinKeys>(m_keygen->relin_keys());
    m_galois_keys = std::make_shared<seal::GaloisKeys>(m_keygen->galois_keys());
  }
  m_public_key = std::make_shared<seal::PublicKey>(m_keygen->public_key());
  m_secret_key = std::make_shared<seal::SecretKey>(m_keygen->secret_key());
  m_encryptor = std::make_shared<seal::Encryptor>(m_context, *m_public_key);
  m_decryptor = std::make_shared<seal::Decryptor>(m_context, *m_secret_key);
  m_evaluator = std::make_shared<seal::Evaluator>(m_context);
  m_ckks_encoder = std::make_shared<seal::CKKSEncoder>(m_context);

  auto coeff_moduli = context_data->parms().coeff_modulus();

  print_encryption_parameters(m_encryption_params, *m_context);

  // Set barrett ratio map
  for (const seal::SmallModulus& modulus : coeff_moduli) {
    const std::uint64_t modulus_value = modulus.value();
    if (modulus_value < (1UL << 31U)) {
      std::array<std::uint64_t, 2> numerator = {0, 1};
      std::array<std::uint64_t, 2> quotient = {0, 0};
      seal::util::divide_uint128_uint64_inplace(numerator.data(), modulus_value,
                                                quotient.data());
      std::uint64_t const_ratio = quotient[0];

      NGRAPH_CHECK(quotient[1] == 0, "Quotient[1] != 0 for modulus");
      m_barrett64_ratio_map[modulus_value] = const_ratio;
    }
  }
}

bool HESealBackend::set_config(const std::map<std::string, std::string>& config,
                               std::string& error) {
  (void)error;  // Avoid unused parameter warning
  NGRAPH_HE_LOG(3) << "Setting config";
  for (const auto& [option, setting] : config) {
    // Check whether client is enabled
    if (option == "enable_client") {
      bool client_enabled = string_to_bool(setting.c_str(), false);
      if (client_enabled) {
        NGRAPH_HE_LOG(3) << "Enabling client from config";
        m_enable_client = true;
      }
    } else if (option == "encryption_parameters") {
      auto new_parms = HESealEncryptionParameters::parse_config_or_use_default(
          setting.c_str());
      update_encryption_parameters(new_parms);
    } else if (option == "enable_gc") {
      m_enable_garbled_circuit = string_to_bool(setting.c_str(), false);
      if (m_enable_garbled_circuit) {
        NGRAPH_HE_LOG(3) << "Enabling garbled circuits from config";
      }
    } else if (option == "mask_gc_inputs") {
      m_mask_gc_inputs = string_to_bool(setting.c_str(), false);
      if (m_mask_gc_inputs) {
        NGRAPH_HE_LOG(3) << "Masking garbled circuits inputs from config";
      } else {
        NGRAPH_HE_LOG(3) << "Not masking garbled circuits inputs from config";
      }
    } else if (option == "mask_gc_outputs") {
      m_mask_gc_outputs = string_to_bool(setting.c_str(), false);
      if (m_mask_gc_outputs) {
        NGRAPH_HE_LOG(3) << "Masking garbled circuits outputs from config";
      } else {
        NGRAPH_HE_LOG(3) << "Not masking garbled circuits outputs from config";
      }
    } else {
      std::string lower_option = to_lower(option);
      std::vector<std::string> lower_settings = split(to_lower(setting), ',');
      // Strip attributes, i.e. "tensor_name:0 => tensor_name"
      std::string tensor_name = option.substr(0, option.find(':', 0));
      m_config_tensors.insert_or_assign(
          tensor_name,
          *HEOpAnnotations::server_plaintext_unpacked_annotation());

      static std::unordered_set<std::string> valid_config_settings{
          "client_input", "encrypt", "packed", ""};
      for (const auto& lower_setting : lower_settings) {
        NGRAPH_CHECK(valid_config_settings.find(lower_setting) !=
                         valid_config_settings.end(),
                     "Invalid config setting ", lower_setting);

        if (lower_setting == "client_input") {
          m_config_tensors.at(tensor_name).set_from_client(true);
        } else if (lower_setting == "encrypt") {
          m_config_tensors.at(tensor_name).set_encrypted(true);
        } else if (lower_setting == "packed") {
          m_config_tensors.at(tensor_name).set_packed(true);
        }
      }
    }
  }

  if (m_enable_garbled_circuit && !m_enable_client) {
    NGRAPH_WARN << "Garbled circuit enabled without enabling client; setting "
                   "Garbled circuit enabled to off";
    m_enable_garbled_circuit = false;
  }

  bool any_config_from_client =
      std::any_of(m_config_tensors.begin(), m_config_tensors.end(),
                  [](const auto& name_and_op_annotation) {
                    const auto& [name, op_annotation] = name_and_op_annotation;
                    return op_annotation.from_client();
                  });
  NGRAPH_CHECK(
      m_enable_client || !any_config_from_client,
      "Configuration specifies client input, but client is not enabled");

  for (const auto& [name, config] : m_config_tensors) {
    NGRAPH_HE_LOG(3) << "Tensor name: " << name << " with config " << config;
  }
  return true;
}

void HESealBackend::update_encryption_parameters(
    const HESealEncryptionParameters& new_parms) {
  if (HESealEncryptionParameters::same_context(m_encryption_params,
                                               new_parms)) {
    m_encryption_params = new_parms;
  } else {
    m_encryption_params = new_parms;
    generate_context();
  }
}

std::shared_ptr<runtime::Tensor> HESealBackend::create_tensor(
    const element::Type& type, const Shape& shape) {
  return create_plain_tensor(type, shape, false);
}

std::shared_ptr<runtime::Tensor> HESealBackend::create_plain_tensor(
    const element::Type& type, const Shape& shape, const bool plaintext_packing,
    const std::string& name) const {
  auto tensor = std::make_shared<HETensor>(
      type, shape, plaintext_packing, complex_packing(), false, *this, name);
  return std::static_pointer_cast<runtime::Tensor>(tensor);
}

std::shared_ptr<runtime::Tensor> HESealBackend::create_cipher_tensor(
    const element::Type& type, const Shape& shape, const bool plaintext_packing,
    const std::string& name) const {
  auto tensor = std::make_shared<HETensor>(
      type, shape, plaintext_packing, complex_packing(), true, *this, name);
  return std::static_pointer_cast<runtime::Tensor>(tensor);
}

std::shared_ptr<runtime::Tensor> HESealBackend::create_packed_cipher_tensor(
    const element::Type& type, const Shape& shape) const {
  auto tensor = std::make_shared<HETensor>(type, shape, true, complex_packing(),
                                           true, *this);
  return std::static_pointer_cast<runtime::Tensor>(tensor);
}

std::shared_ptr<runtime::Tensor> HESealBackend::create_packed_plain_tensor(
    const element::Type& type, const Shape& shape) const {
  auto tensor = std::make_shared<HETensor>(type, shape, true, complex_packing(),
                                           false, *this);
  return std::static_pointer_cast<runtime::Tensor>(tensor);
}

// NOLINTNEXTLINE
std::shared_ptr<runtime::Executable> HESealBackend::compile(
    std::shared_ptr<Function> function, bool enable_performance_data) {
  NGRAPH_HE_LOG(1) << "Compiling function with "
                   << function->get_parameters().size() << " parameters";

  for (auto& param : function->get_parameters()) {
    NGRAPH_HE_LOG(3) << "Compiling function with parameter name "
                     << param->get_name() << " (" << param->get_shape() << ")";

    for (const auto& tag : param->get_provenance_tags()) {
      NGRAPH_HE_LOG(3) << "Tag " << tag;
    }
  }

  for (auto& param : function->get_parameters()) {
    auto it =
        std::find_if(m_config_tensors.begin(), m_config_tensors.end(),
                     [&](const auto& config) {
                       const auto& [tensor_name, annotation] = config;
                       return param_originates_from_name(*param, tensor_name);
                     });
    if (it != m_config_tensors.end()) {
      const auto& [tensor_name, annotation] = *it;
      param->set_op_annotations(std::make_shared<HEOpAnnotations>(annotation));
    } else {
      param->set_op_annotations(
          HEOpAnnotations::server_plaintext_unpacked_annotation());
    }
  }

  return std::dynamic_pointer_cast<runtime::Executable>(
      std::make_shared<HESealExecutable>(function, enable_performance_data,
                                         *this));
}

bool HESealBackend::is_supported(const Node& node) const {
  return m_unsupported_op_name_list.find(node.description()) ==
             m_unsupported_op_name_list.end() &&
         is_supported_type(node.get_element_type());
}

void HESealBackend::encrypt(std::shared_ptr<SealCiphertextWrapper>& output,
                            const HEPlaintext& input, const element::Type& type,
                            bool complex_packing) const {
  NGRAPH_CHECK(!input.empty(), "Input has no values in encrypt");
  ngraph::runtime::he::encrypt(output, input, m_context->first_parms_id(), type,
                               get_scale(), *m_ckks_encoder, *m_encryptor,
                               complex_packing);
}

void HESealBackend::decrypt(HEPlaintext& output,
                            const SealCiphertextWrapper& input,
                            size_t batch_size,
                            const bool complex_packing) const {
  ngraph::runtime::he::decrypt(output, input, complex_packing, *m_decryptor,
                               *m_ckks_encoder, m_context, batch_size);
}

}  // namespace ngraph::runtime::he
