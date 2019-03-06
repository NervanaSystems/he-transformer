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

#include "he_cipher_tensor.hpp"
#include "he_plain_tensor.hpp"
#include "he_tensor.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "seal/bfv/he_seal_bfv_backend.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_parameter.hpp"
#include "seal/he_seal_util.hpp"
#include "seal/seal.h"

using namespace ngraph;
using namespace std;

const static runtime::he::he_seal::HESealParameter
parse_seal_bfv_config_or_use_default() {
  try {
    const char* config_path = getenv("NGRAPH_HE_SEAL_CONFIG");
    if (config_path != nullptr) {
      // Read file to string
      ifstream f(config_path);
      stringstream ss;
      ss << f.rdbuf();
      string s = ss.str();

      // Parse json
      nlohmann::json js = nlohmann::json::parse(s);
      string scheme_name = js["scheme_name"];
      uint64_t poly_modulus_degree = js["poly_modulus_degree"];
      uint64_t plain_modulus = js["plain_modulus"];
      uint64_t security_level = js["security_level"];
      uint64_t evaluation_decomposition_bit_count =
          js["evaluation_decomposition_bit_count"];

      NGRAPH_INFO << "Using SEAL BFV config for parameters: " << config_path;
      return runtime::he::he_seal::HESealParameter(
          scheme_name, poly_modulus_degree, plain_modulus, security_level,
          evaluation_decomposition_bit_count);
    } else {
      NGRAPH_INFO << "Using SEAL BFV default parameters" << config_path;
      throw runtime_error("config_path is NULL");
    }
  } catch (const exception& e) {
    return runtime::he::he_seal::HESealParameter(
        "HE_SEAL_BFV",  // scheme name
        4096,           // poly_modulus_degree
        1 << 10,        // plain_modulus
        128,            // security_level
        16              // evaluation_decomposition_bit_count
    );
  }
}

const static runtime::he::he_seal::HESealParameter default_seal_bfv_parameter =
    parse_seal_bfv_config_or_use_default();

runtime::he::he_seal::HESealBFVBackend::HESealBFVBackend()
    : runtime::he::he_seal::HESealBFVBackend(
          make_shared<runtime::he::he_seal::HESealParameter>(
              default_seal_bfv_parameter)) {}

runtime::he::he_seal::HESealBFVBackend::HESealBFVBackend(
    const shared_ptr<runtime::he::he_seal::HESealParameter>& sp) {
  assert_valid_seal_bfv_parameter(sp);

  m_context = make_seal_context(sp);
  print_seal_context(*m_context);

  auto context_data = m_context->context_data();

  // Keygen, encryptor and decryptor
  m_keygen = make_shared<seal::KeyGenerator>(m_context);
  m_relin_keys = make_shared<seal::RelinKeys>(
      m_keygen->relin_keys(sp->m_evaluation_decomposition_bit_count));
  m_public_key = make_shared<seal::PublicKey>(m_keygen->public_key());
  m_secret_key = make_shared<seal::SecretKey>(m_keygen->secret_key());
  m_encryptor = make_shared<seal::Encryptor>(m_context, *m_public_key);
  m_decryptor = make_shared<seal::Decryptor>(m_context, *m_secret_key);

  // Evaluator
  m_evaluator = make_shared<seal::Evaluator>(m_context);

  // Encoders
  if (context_data->qualifiers().using_batching) {
    m_batch_encoder = make_shared<seal::BatchEncoder>(m_context);
  } else {
    NGRAPH_WARN << "BFV encryption parameters not valid for batching";
  }
  m_integer_encoder = make_shared<seal::IntegerEncoder>(m_context);

  // Plaintext constants
  m_plaintext_map[-1] =
      make_shared<SealPlaintextWrapper>(m_integer_encoder->encode(-1));
  m_plaintext_map[0] =
      make_shared<SealPlaintextWrapper>(m_integer_encoder->encode(0));
  m_plaintext_map[1] =
      make_shared<SealPlaintextWrapper>(m_integer_encoder->encode(1));
}

extern "C" runtime::Backend* new_bfv_backend(const char* configuration_string) {
  return new runtime::he::he_seal::HESealBFVBackend();
}

shared_ptr<seal::SEALContext>
runtime::he::he_seal::HESealBFVBackend::make_seal_context(
    const shared_ptr<runtime::he::he_seal::HESealParameter> sp) {
  seal::EncryptionParameters parms =
      (sp->m_scheme_name == "HE_SEAL_BFV"
           ? seal::scheme_type::BFV
           : throw ngraph_error("Invalid scheme name \"" + sp->m_scheme_name +
                                "\""));

  parms.set_poly_modulus_degree(sp->m_poly_modulus_degree);

  if (sp->m_security_level == 128) {
    parms.set_coeff_modulus(
        seal::DefaultParams::coeff_modulus_128(sp->m_poly_modulus_degree));
  } else if (sp->m_security_level == 192) {
    parms.set_coeff_modulus(
        seal::DefaultParams::coeff_modulus_192(sp->m_poly_modulus_degree));
  } else if (sp->m_security_level == 256) {
    parms.set_coeff_modulus(
        seal::DefaultParams::coeff_modulus_256(sp->m_poly_modulus_degree));
  } else {
    throw ngraph_error("sp.security_level must be 128, 192, or 256");
  }
  parms.set_plain_modulus(sp->m_plain_modulus);

  return seal::SEALContext::Create(parms);
}

namespace {
static class HESealBFVStaticInit {
 public:
  HESealBFVStaticInit() {
    runtime::BackendManager::register_backend("HE_SEAL_BFV", new_bfv_backend);
  }
  ~HESealBFVStaticInit() {}
} s_he_seal_bfv_static_init;
}  // namespace

void runtime::he::he_seal::HESealBFVBackend::assert_valid_seal_bfv_parameter(
    const shared_ptr<runtime::he::he_seal::HESealParameter>& sp) const {
  assert_valid_seal_parameter(sp);
  if (sp->m_scheme_name != "HE_SEAL_BFV") {
    throw ngraph_error("Invalid scheme name");
  }
}

shared_ptr<runtime::Tensor>
runtime::he::he_seal::HESealBFVBackend::create_batched_cipher_tensor(
    const element::Type& element_type, const Shape& shape) {
  throw ngraph_error("HESealBFVBackend::create_batched_tensor unimplemented");
}

shared_ptr<runtime::Tensor>
runtime::he::he_seal::HESealBFVBackend::create_batched_plain_tensor(
    const element::Type& element_type, const Shape& shape) {
  throw ngraph_error("HESealBFVBackend::create_batched_tensor unimplemented");
}

void runtime::he::he_seal::HESealBFVBackend::encode(
    shared_ptr<runtime::he::HEPlaintext>& output, const void* input,
    const element::Type& element_type, size_t count) const {
  if (count != 1) {
    throw ngraph_error("Batching not enabled for SEAL in encode");
  }
  const string type_name = element_type.c_type_string();

  if (type_name == "float") {
    double value = (double)(*(float*)input);
    if (m_plaintext_map.find(value) != m_plaintext_map.end()) {
      auto plain_value =
          static_pointer_cast<const runtime::he::he_seal::SealPlaintextWrapper>(
              get_valued_plaintext(value));
      output =
          make_shared<runtime::he::he_seal::SealPlaintextWrapper>(*plain_value);
    } else {
      float float_val = *(float*)input;
      int32_t int_val;
      if (ceilf(float_val) == float_val) {
        int_val = static_cast<int32_t>(float_val);
      } else {
        NGRAPH_INFO << "BFV float only supported for int32_t";
        throw ngraph_error("BFV float only supported for int32_t");
      }

      output = make_shared<runtime::he::he_seal::SealPlaintextWrapper>(
          m_integer_encoder->encode(int_val));
    }
  } else {
    NGRAPH_INFO << "Unsupported element type in decode " << type_name;
    throw ngraph_error("Unsupported element type " + type_name);
  }
}

void runtime::he::he_seal::HESealBFVBackend::decode(
    void* output, const runtime::he::HEPlaintext* input,
    const element::Type& element_type, size_t count) const {
  if (count != 1) {
    throw ngraph_error("Batching not enabled for SEAL in decode");
  }
  const string type_name = element_type.c_type_string();

  if (auto seal_input = dynamic_cast<const SealPlaintextWrapper*>(input)) {
    if (type_name == "float") {
      int32_t val = m_integer_encoder->decode_int32(seal_input->m_plaintext);
      float fl_val{val};
      memcpy(output, &fl_val, element_type.size());
    } else {
      NGRAPH_INFO << "Unsupported element type in decode " << type_name;
      throw ngraph_error("Unsupported element type " + type_name);
    }
  } else {
    throw ngraph_error("HESealBFVBackend::decode input is not seal plaintext");
  }
}

void runtime::he::he_seal::HESealBFVBackend::handle_message(
    const runtime::he::TCPMessage& message) {
  throw ngraph_error("handle_message not implemented in BFV backend");
}