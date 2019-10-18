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

#include <algorithm>
#include <limits>
#include <memory>

#include "he_op_annotations.hpp"
#include "logging/ngraph_he_log.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_executable.hpp"
#include "seal/seal.h"
#include "seal/seal_util.hpp"

using json = nlohmann::json;

extern "C" ngraph::runtime::BackendConstructor*
get_backend_constructor_pointer() {
  class HESealBackendConstructor : public ngraph::runtime::BackendConstructor {
   public:
    std::shared_ptr<ngraph::runtime::Backend> create(
        const std::string& config) override {
      NGRAPH_HE_LOG(5) << "Creating backend with config string " << config;
      return std::make_shared<ngraph::he::HESealBackend>();
    }
  };

  static std::unique_ptr<ngraph::runtime::BackendConstructor>
      s_backend_constructor(new HESealBackendConstructor());
  return s_backend_constructor.get();
}

namespace ngraph {
namespace he {

HESealBackend::HESealBackend()
    : HESealBackend(HESealEncryptionParameters::parse_config_or_use_default(
          getenv("NGRAPH_HE_SEAL_CONFIG"))) {}

HESealBackend::HESealBackend(const HESealEncryptionParameters& parms)
    : m_encryption_params(parms) {
  generate_context();
}

void HESealBackend::generate_context() {
  seal::sec_level_type sec_level =
      seal_security_level(m_encryption_params.security_level());

  m_context = seal::SEALContext::Create(
      m_encryption_params.seal_encryption_parameters(), true, sec_level);

  auto context_data = m_context->key_context_data();

  m_keygen = std::make_shared<seal::KeyGenerator>(m_context);
  m_relin_keys = std::make_shared<seal::RelinKeys>(m_keygen->relin_keys());
  m_galois_keys = std::make_shared<seal::GaloisKeys>(m_keygen->galois_keys());
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
    if (modulus_value < (1UL << 31)) {
      std::uint64_t numerator[3]{0, 1};
      std::uint64_t quotient[3]{0, 0};
      seal::util::divide_uint128_uint64_inplace(numerator, modulus_value,
                                                quotient);
      std::uint64_t const_ratio = quotient[0];

      NGRAPH_CHECK(quotient[1] == 0, "Quotient[1] != 0 for modulus");
      m_barrett64_ratio_map[modulus_value] = const_ratio;
    }
  }
}

bool HESealBackend::set_config(const std::map<std::string, std::string>& config,
                               std::string& error) {
  NGRAPH_HE_LOG(3) << "Setting config";
  for (const auto& config_opt : config) {
    std::string option = ngraph::to_lower(config_opt.first);
    std::string setting = config_opt.second;
    std::vector<std::string> lower_settings =
        ngraph::split(ngraph::to_lower(config_opt.second), ',');
    // Strip attributes, i.e. "tensor_name:0 => tensor_name"
    std::string tensor_name = option.substr(0, option.find(":", 0));

    for (const auto& lower_setting : lower_settings) {
      if (lower_setting == "client_input") {
        m_client_tensor_names.insert(tensor_name);
      } else if (lower_setting == "encrypt") {
        m_encrypted_tensor_names.insert(tensor_name);
      } else if (lower_setting == "plain") {
        m_plaintext_tensor_names.insert(tensor_name);
      } else if (lower_setting == "packed") {
        m_packed_tensor_names.insert(tensor_name);
      }
    }

    // Check whether client is enabled
    if (option == "enable_client") {
      bool client_enabled = flag_to_bool(setting.c_str(), false);
      if (client_enabled) {
        NGRAPH_HE_LOG(3) << "Enabling client from config";
        m_enable_client = true;
      }
    } else if (option == "encryption_parameters") {
      auto new_parms = HESealEncryptionParameters::parse_config_or_use_default(
          setting.c_str());
      update_encryption_parameters(new_parms);
    }
  }

  if (m_client_tensor_names.size() > 0 && !m_enable_client) {
    NGRAPH_WARN
        << "Configuration specifies client input, but client is not enabled";
    m_client_tensor_names.clear();
  }

  for (const auto& tensor_name : m_client_tensor_names) {
    NGRAPH_HE_LOG(3) << "Client tensor name " << tensor_name;
  }
  for (const auto& tensor_name : m_encrypted_tensor_names) {
    NGRAPH_HE_LOG(3) << "Encrypted tensor name " << tensor_name;
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

std::shared_ptr<ngraph::runtime::Tensor> HESealBackend::create_tensor(
    const element::Type& type, const Shape& shape) {
  return create_plain_tensor(type, shape, false);
}

std::shared_ptr<ngraph::runtime::Tensor> HESealBackend::create_plain_tensor(
    const element::Type& type, const Shape& shape, const bool plaintext_packing,
    const std::string& name) const {
  auto tensor = std::make_shared<HETensor>(
      type, shape, plaintext_packing, complex_packing(), false, *this, name);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(tensor);
}

std::shared_ptr<ngraph::runtime::Tensor> HESealBackend::create_cipher_tensor(
    const element::Type& type, const Shape& shape, const bool plaintext_packing,
    const std::string& name) const {
  auto tensor = std::make_shared<HETensor>(
      type, shape, plaintext_packing, complex_packing(), true, *this, name);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(tensor);
}

std::shared_ptr<ngraph::runtime::Tensor>
HESealBackend::create_packed_cipher_tensor(const element::Type& type,
                                           const Shape& shape) const {
  auto tensor = std::make_shared<HETensor>(type, shape, true, complex_packing(),
                                           true, *this);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(tensor);
}

std::shared_ptr<ngraph::runtime::Tensor>
HESealBackend::create_packed_plain_tensor(const element::Type& type,
                                          const Shape& shape) const {
  auto tensor = std::make_shared<HETensor>(type, shape, true, complex_packing(),
                                           false, *this);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(tensor);
}

std::shared_ptr<ngraph::runtime::Executable> HESealBackend::compile(
    std::shared_ptr<Function> function, bool enable_performance_collection) {
  auto from_client_annotation =
      std::make_shared<HEOpAnnotations>(true, false, false);

  NGRAPH_HE_LOG(1) << "Compiling function with "
                   << function->get_parameters().size() << " parameters";

  for (auto& param : function->get_parameters()) {
    NGRAPH_HE_LOG(3) << "Compiling function with parameter name "
                     << param->get_name() << " (" << param->get_shape() << ")";

    for (const auto& tag : param->get_provenance_tags()) {
      NGRAPH_HE_LOG(3) << "Tag " << tag;
    }
  }

  for (const auto& name : get_client_tensor_names()) {
    NGRAPH_HE_LOG(3) << "get_client_tensor_names " << name;
    bool matching_param = false;
    bool has_tag = false;
    for (auto& param : function->get_parameters()) {
      has_tag |= (param->get_provenance_tags().size() != 0);

      if (param_originates_from_name(*param, name)) {
        NGRAPH_HE_LOG(3) << "Setting tensor name " << param->get_name() << " ("
                         << param->get_shape() << ") as from client";
        param->set_op_annotations(from_client_annotation);
        matching_param = true;
      }
    }
    // ngraph-bridge calls compile() twice, once before adding tags, and once
    // after
    NGRAPH_CHECK(!has_tag || matching_param, "Function has no parameter named ",
                 name);
  }

  NGRAPH_HE_LOG(3) << "Setting encrypted tags";
  for (const auto& name : get_encrypted_tensor_names()) {
    for (auto& param : function->get_parameters()) {
      if (param_originates_from_name(*param, name)) {
        NGRAPH_HE_LOG(5) << "Setting tensor name " << param->get_name()
                         << param->get_shape() << ") as encrypted";
        auto current_annotation = std::dynamic_pointer_cast<HEOpAnnotations>(
            param->get_op_annotations());
        if (current_annotation == nullptr) {
          param->set_op_annotations(
              HEOpAnnotations::server_ciphertext_unpacked_annotation());
        } else {
          current_annotation->set_encrypted(true);
        }
        NGRAPH_HE_LOG(5) << "Set tensor name " << param->get_name() << " ("
                         << param->get_shape() << ") as encrypted";
      }
    }
  }

  NGRAPH_HE_LOG(3) << "Setting plaintext tags";
  for (const auto& name : get_plaintext_tensor_names()) {
    NGRAPH_HE_LOG(5) << "Plaintext tensor name " << name;
    for (auto& param : function->get_parameters()) {
      if (param_originates_from_name(*param, name)) {
        NGRAPH_HE_LOG(3) << "Setting tensor name " << param->get_name() << " ("
                         << param->get_shape() << ") as plaintext";
        auto current_annotation = std::dynamic_pointer_cast<HEOpAnnotations>(
            param->get_op_annotations());
        if (current_annotation == nullptr) {
          param->set_op_annotations(
              HEOpAnnotations::server_plaintext_unpacked_annotation());
        } else {
          current_annotation->set_encrypted(false);
        }
        NGRAPH_HE_LOG(5) << "Set tensor name " << param->get_name() << " ("
                         << param->get_shape() << ") as plaintext";
      }
    }
  }

  NGRAPH_HE_LOG(3) << "Setting packed tags";
  for (const auto& name : get_packed_tensor_names()) {
    NGRAPH_HE_LOG(5) << "Packed tensor name " << name;
    for (auto& param : function->get_parameters()) {
      if (param_originates_from_name(*param, name)) {
        NGRAPH_HE_LOG(3) << "Setting tensor name " << param->get_name() << " ("
                         << param->get_shape() << ") as packed";
        auto current_annotation = std::dynamic_pointer_cast<HEOpAnnotations>(
            param->get_op_annotations());
        if (current_annotation == nullptr) {
          param->set_op_annotations(
              HEOpAnnotations::server_plaintext_packed_annotation());
        } else {
          current_annotation->set_packed(true);
        }
        NGRAPH_HE_LOG(5) << "Set tensor name " << param->get_name() << " ("
                         << param->get_shape() << ") as plaintext";
      }
    }
  }

  return std::make_shared<HESealExecutable>(
      function, enable_performance_collection, *this, m_enable_client);
}

bool HESealBackend::is_supported(const ngraph::Node& node) const {
  return false;
  /* return m_unsupported_op_name_list.find(node.description()) ==
             m_unsupported_op_name_list.end() &&
         is_supported_type(node.get_type()); */
}

std::shared_ptr<SealCiphertextWrapper> HESealBackend::create_valued_ciphertext(
    float value, const element::Type& type, size_t batch_size) const {
  NGRAPH_CHECK(type == element::f32, "element type ", type, "unsupported");
  if (batch_size != 1) {
    throw ngraph_error(
        "HESealBackend::create_valued_ciphertext only supports batch size 1");
  }
  auto plaintext = HEPlaintext({value});
  auto ciphertext = create_empty_ciphertext();

  encrypt(ciphertext, plaintext, type, complex_packing());
  return ciphertext;
}

void HESealBackend::encrypt(std::shared_ptr<SealCiphertextWrapper>& output,
                            const HEPlaintext& input, const element::Type& type,
                            bool complex_packing) const {
  auto plaintext = SealPlaintextWrapper(complex_packing);

  NGRAPH_CHECK(input.size() > 0, "Input has no values in encrypt");
  ngraph::he::encrypt(output, input, m_context->first_parms_id(), type,
                      get_scale(), *m_ckks_encoder, *m_encryptor,
                      complex_packing);
}

void HESealBackend::decrypt(HEPlaintext& output,
                            const SealCiphertextWrapper& input) const {
  ngraph::he::decrypt(output, input, *m_decryptor, *m_ckks_encoder);
}

void HESealBackend::decode(HEPlaintext& output,
                           const SealPlaintextWrapper& input) const {
  NGRAPH_CHECK(false, "decode unimplemented");
  // ngraph::he::decode(output, input, *m_ckks_encoder);
}

}  // namespace he
}  // namespace ngraph
