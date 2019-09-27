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

#include <limits>
#include <memory>

#include "he_plain_tensor.hpp"
#include "he_seal_cipher_tensor.hpp"
#include "logging/ngraph_he_log.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/util.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_executable.hpp"
#include "seal/seal.h"
#include "seal/seal_util.hpp"

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

ngraph::he::HESealBackend::HESealBackend()
    : ngraph::he::HESealBackend(
          ngraph::he::HESealEncryptionParameters::parse_config_or_use_default(
              getenv("NGRAPH_HE_SEAL_CONFIG"))) {}

ngraph::he::HESealBackend::HESealBackend(
    const ngraph::he::HESealEncryptionParameters& parms)
    : m_encryption_params(parms) {
  seal::sec_level_type sec_level =
      ngraph::he::seal_security_level(parms.security_level());

  m_context = seal::SEALContext::Create(parms.seal_encryption_parameters(),
                                        true, sec_level);

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
  if (parms.scale() == 0) {
    m_scale = ngraph::he::choose_scale(coeff_moduli);
  } else {
    m_scale = parms.scale();
  }

  m_encryption_params.scale() = m_scale;

  if (m_encrypt_all_params || m_encrypt_parameter_shapes.size() != 0 ||
      m_encrypt_model) {
    print_encryption_parameters(m_encryption_params, *m_context);
  }

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

  NGRAPH_CHECK(!(m_encrypt_model && complex_packing()),
               "Encrypting the model is incompatible with complex packing");
}

bool ngraph::he::HESealBackend::set_config(
    const std::map<std::string, std::string>& config, std::string& error) {
  /// \brief Checks if a string description contains a shape=(x,y,z) substring
  /// \param[in] description input string
  /// \param[out] shape_str Resulting string description of shape if parsed
  /// \returns whether or not the string contains a shape description
  auto parse_shape = [](const std::string& description,
                        std::string& shape_str) {
    if (description.substr(0, 6) == "Tensor") {
      static std::string shape_start_str{"shape=("};
      size_t shape_start = description.find(shape_start_str);
      size_t shape_end = description.find(")", shape_start);
      if (shape_start == std::string::npos || shape_end == std::string::npos) {
        NGRAPH_WARN << "Parameter " << description
                    << " is tensor, but has no shape";
        return false;
      }
      std::string tensor_shape_str =
          description.substr(shape_start + shape_start_str.size(),
                             shape_end - shape_start - shape_start_str.size());

      std::vector<std::string> dimensions_str =
          ngraph::split(tensor_shape_str, ',', true);

      shape_str = ngraph::join(dimensions_str, "x");
      NGRAPH_HE_LOG(5) << "shape_str " << shape_str;

      return true;
    } else {
      return false;
    }
  };

  for (const auto& config_opt : config) {
    // Check for encryption of parameters
    std::string shape_str;
    if (ngraph::to_lower(config_opt.second) == "encrypt" &&
        parse_shape(config_opt.first, shape_str)) {
      m_encrypt_parameter_shapes.insert(shape_str);
    }

    // Check whether client is enabled
    if (ngraph::to_lower(config_opt.first) == "enable_client") {
      bool client_enabled =
          ngraph::he::flag_to_bool(config_opt.second.c_str(), false);
      if (client_enabled) {
        NGRAPH_HE_LOG(3) << "Enabling client from config";
        m_enable_client = true;
      }
    }
  }

  for (const auto& shape : m_encrypt_parameter_shapes) {
    NGRAPH_HE_LOG(3) << "Encryption parameter shape " << shape;
  }
  return true;
}

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HESealBackend::create_tensor(const element::Type& element_type,
                                         const Shape& shape) {
  if (pack_data()) {
    return create_packed_plain_tensor(element_type, shape);
  } else {
    return create_plain_tensor(element_type, shape);
  }
}

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HESealBackend::create_plain_tensor(
    const element::Type& element_type, const Shape& shape,
    const bool packed) const {
  auto rc =
      std::make_shared<ngraph::he::HEPlainTensor>(element_type, shape, packed);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(rc);
}

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HESealBackend::create_cipher_tensor(
    const element::Type& element_type, const Shape& shape, const bool packed,
    const std::string& name) const {
  auto rc = std::make_shared<ngraph::he::HESealCipherTensor>(
      element_type, shape, *this, packed, name);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(rc);
}

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HESealBackend::create_packed_cipher_tensor(
    const element::Type& type, const Shape& shape) {
  auto rc = std::make_shared<ngraph::he::HESealCipherTensor>(type, shape, *this,
                                                             true);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(rc);
}

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HESealBackend::create_packed_plain_tensor(const element::Type& type,
                                                      const Shape& shape) {
  auto rc = std::make_shared<ngraph::he::HEPlainTensor>(type, shape, true);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(rc);
}

std::shared_ptr<ngraph::runtime::Executable> ngraph::he::HESealBackend::compile(
    std::shared_ptr<Function> function, bool enable_performance_collection) {
  return std::make_shared<HESealExecutable>(
      function, enable_performance_collection, *this,
      m_encrypt_parameter_shapes, m_encrypt_all_params, m_encrypt_model,
      pack_data(), complex_packing(), m_enable_client);
}

bool ngraph::he::HESealBackend::is_supported(const ngraph::Node& node) const {
  return m_unsupported_op_name_list.find(node.description()) ==
             m_unsupported_op_name_list.end() &&
         is_supported_type(node.get_element_type());
}

std::shared_ptr<ngraph::he::SealCiphertextWrapper>
ngraph::he::HESealBackend::create_valued_ciphertext(
    float value, const element::Type& element_type, size_t batch_size) const {
  NGRAPH_CHECK(element_type == element::f32, "element type ", element_type,
               "unsupported");
  if (batch_size != 1) {
    throw ngraph_error(
        "HESealBackend::create_valued_ciphertext only supports batch size 1");
  }
  auto plaintext = HEPlaintext(value);
  auto ciphertext = create_empty_ciphertext();

  encrypt(ciphertext, plaintext, element_type, complex_packing());
  return ciphertext;
}

void ngraph::he::HESealBackend::encrypt(
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& output,
    const ngraph::he::HEPlaintext& input, const element::Type& element_type,
    bool complex_packing) const {
  auto plaintext = SealPlaintextWrapper(complex_packing);

  NGRAPH_CHECK(input.num_values() > 0, "Input has no values in encrypt");
  ngraph::he::encrypt(output, input, m_context->first_parms_id(), element_type,
                      m_scale, *m_ckks_encoder, *m_encryptor, complex_packing);
}

void ngraph::he::HESealBackend::decrypt(
    ngraph::he::HEPlaintext& output,
    const ngraph::he::SealCiphertextWrapper& input) const {
  ngraph::he::decrypt(output, input, *m_decryptor, *m_ckks_encoder);
}

void ngraph::he::HESealBackend::decode(
    ngraph::he::HEPlaintext& output,
    const ngraph::he::SealPlaintextWrapper& input) const {
  ngraph::he::decode(output, input, *m_ckks_encoder);
}
