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
#include "ngraph/runtime/backend_manager.hpp"
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
      return std::make_shared<ngraph::he::HESealBackend>();
    }
  };

  static std::unique_ptr<ngraph::runtime::BackendConstructor>
      s_backend_constructor(new HESealBackendConstructor());
  return s_backend_constructor.get();
}

ngraph::he::HESealBackend::HESealBackend()
    : ngraph::he::HESealBackend(
          ngraph::he::parse_config_or_use_default("HE_SEAL")) {}

ngraph::he::HESealBackend::HESealBackend(
    const ngraph::he::HESealEncryptionParameters& parms)
    : m_encryption_params(parms) {
  seal::sec_level_type sec_level = seal::sec_level_type::none;
  if (parms.security_level() == 128) {
    sec_level = seal::sec_level_type::tc128;
  } else if (parms.security_level() == 192) {
    sec_level = seal::sec_level_type::tc192;
  } else if (parms.security_level() == 256) {
    sec_level = seal::sec_level_type::tc256;
  } else if (parms.security_level() == 0) {
    if (m_encrypt_data || m_encrypt_model) {
      NGRAPH_WARN
          << "Parameter selection does not enforce minimum security level";
    }
  } else {
    throw ngraph_error("Invalid security level");
  }

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

  if (m_encrypt_data) {
    print_seal_context(*m_context);
    NGRAPH_HE_LOG(1) << "Scale " << m_scale;
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

  NGRAPH_CHECK(!(m_encrypt_model && m_complex_packing),
               "NGRAPH_ENCRYPT_MODEL is incompatible with NGRAPH_COMPLEX_PACK");
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
  auto rc = std::make_shared<ngraph::he::HEPlainTensor>(element_type, shape,
                                                        *this, packed);
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
  auto rc =
      std::make_shared<ngraph::he::HEPlainTensor>(type, shape, *this, true);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(rc);
}

std::shared_ptr<ngraph::runtime::Executable> ngraph::he::HESealBackend::compile(
    std::shared_ptr<Function> function, bool enable_performance_collection) {
  return std::make_shared<HESealExecutable>(
      function, enable_performance_collection, *this, m_encrypt_data,
      m_encrypt_model, pack_data(), m_complex_packing, m_enable_client);
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
