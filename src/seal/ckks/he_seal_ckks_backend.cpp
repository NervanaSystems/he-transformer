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
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_parameter.hpp"
#include "seal/he_seal_util.hpp"
#include "tcp/tcp_message.hpp"

#include "seal/seal.h"

using namespace ngraph;
using namespace std;

const static runtime::he::he_seal::HESealParameter
parse_seal_ckks_config_or_use_default() {
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
      uint64_t security_level = js["security_level"];
      uint64_t evaluation_decomposition_bit_count =
          js["evaluation_decomposition_bit_count"];

      auto coeff_mod = js.find("coeff_modulus");
      if (coeff_mod != js.end()) {
        string coeff_mod_name = coeff_mod->begin().key();

        set<string> valid_coeff_mods{"small_mods_30bit", "small_mods_40bit",
                                     "small_mods_50bit", "small_mods_60bit"};

        auto valid_coeff_mod = valid_coeff_mods.find(coeff_mod_name);
        if (valid_coeff_mod == valid_coeff_mods.end()) {
          throw ngraph_error("Coeff modulus " + coeff_mod_name + " not valid");
        }
        std::uint64_t bit_count = std::stoi(coeff_mod_name.substr(11, 2));
        std::uint64_t coeff_count = coeff_mod->begin().value();

        NGRAPH_INFO << "Using SEAL CKKS config with " << coeff_count << " "
                    << bit_count << "-bit coefficients";

        runtime::he::he_seal::HESealParameter::CoeffModulus coeff_modulus{
            bit_count, coeff_count};

        return runtime::he::he_seal::HESealParameter(
            scheme_name, poly_modulus_degree, security_level,
            evaluation_decomposition_bit_count, coeff_modulus);
      }

      NGRAPH_INFO << "Using SEAL CKKS config for parameters: " << config_path;
      return runtime::he::he_seal::HESealParameter(
          scheme_name, poly_modulus_degree, security_level,
          evaluation_decomposition_bit_count);
    } else {
      NGRAPH_INFO << "Using SEAL CKKS default parameters" << config_path;
      throw runtime_error("config_path is NULL");
    }
  } catch (const exception& e) {
    NGRAPH_INFO << "Error " << e.what();
    NGRAPH_INFO << "Error using NGRAPH_HE_SEAL_CONFIG. Using default ";
    return runtime::he::he_seal::HESealParameter(
        "HE_SEAL_CKKS",  // scheme name
        1024,            // poly_modulus_degree
        128,             // security_level
        60,              // evaluation_decomposition_bit_count
                         // Coefficient modulus
        runtime::he::he_seal::HESealParameter::CoeffModulus{30, 4});
  }
}

const static runtime::he::he_seal::HESealParameter default_seal_ckks_parameter =
    parse_seal_ckks_config_or_use_default();

runtime::he::he_seal::HESealCKKSBackend::HESealCKKSBackend()
    : runtime::he::he_seal::HESealCKKSBackend(
          make_shared<runtime::he::he_seal::HESealParameter>(
              default_seal_ckks_parameter)) {}

runtime::he::he_seal::HESealCKKSBackend::HESealCKKSBackend(
    const shared_ptr<runtime::he::he_seal::HESealParameter>& sp) {
  assert_valid_seal_ckks_parameter(sp);

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

  m_scale =
      static_cast<double>(context_data->parms().coeff_modulus().back().value());

  // Encoder
  m_ckks_encoder = make_shared<seal::CKKSEncoder>(m_context);

  // Plaintext constants
  shared_ptr<runtime::he::HEPlaintext> plaintext_neg1 =
      create_empty_plaintext();
  shared_ptr<runtime::he::HEPlaintext> plaintext_0 = create_empty_plaintext();
  shared_ptr<runtime::he::HEPlaintext> plaintext_1 = create_empty_plaintext();

  m_ckks_encoder->encode(
      -1, m_scale,
      dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(
          plaintext_neg1)
          ->m_plaintext);

  m_ckks_encoder->encode(
      0, m_scale,
      dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(
          plaintext_0)
          ->m_plaintext);

  m_ckks_encoder->encode(
      0, m_scale,
      dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(
          plaintext_0)
          ->m_plaintext,
      seal::MemoryPoolHandle::ThreadLocal());

  /* m_plaintext_map[-1] = plaintext_neg1;
   m_plaintext_map[0] = plaintext_0;
   m_plaintext_map[1] = plaintext_1;*/

  // Start server
  sleep(1);
  NGRAPH_INFO << "Starting CKKS server";
  start_server();
  NGRAPH_INFO << "Started CKKS server";

  /*
  std::stringstream param_stream;
  seal::EncryptionParameters::Save(*m_encryption_parms, param_stream);
  auto context_message =
      TCPMessage(MessageType::encryption_parameters, param_stream);

  // Send
  NGRAPH_INFO << "Server about to write message";
  m_tcp_server->write_message(context_message);
  NGRAPH_INFO << "Server wrote message";
  */
}

extern "C" runtime::Backend* new_ckks_backend(
    const char* configuration_string) {
  return new runtime::he::he_seal::HESealCKKSBackend();
}

shared_ptr<seal::SEALContext>
runtime::he::he_seal::HESealCKKSBackend::make_seal_context(
    const shared_ptr<runtime::he::he_seal::HESealParameter> sp) {
  auto encryption_parms =
      make_shared<seal::EncryptionParameters>(seal::scheme_type::CKKS);

  if (sp->m_scheme_name != "HE_SEAL_CKKS") {
    throw ngraph_error("Invalid scheme name \"" + sp->m_scheme_name + "\"");
  }

  encryption_parms->set_poly_modulus_degree(sp->m_poly_modulus_degree);

  bool custom_coeff_modulus = (sp->m_coeff_modulus.bit_count != 0);

  if (custom_coeff_modulus) {
    if (sp->m_coeff_modulus.bit_count == 30) {
      std::vector<seal::SmallModulus> small_mods_30_bit =
          seal::util::global_variables::default_small_mods_30bit;
      encryption_parms->set_coeff_modulus(
          {small_mods_30_bit.begin(),
           small_mods_30_bit.begin() + sp->m_coeff_modulus.coeff_count});
    } else if (sp->m_coeff_modulus.bit_count == 40) {
      std::vector<seal::SmallModulus> small_mods_40_bit =
          seal::util::global_variables::default_small_mods_40bit;
      encryption_parms->set_coeff_modulus(
          {small_mods_40_bit.begin(),
           small_mods_40_bit.begin() + sp->m_coeff_modulus.coeff_count});
    } else if (sp->m_coeff_modulus.bit_count == 50) {
      std::vector<seal::SmallModulus> small_mods_50_bit =
          seal::util::global_variables::default_small_mods_50bit;
      encryption_parms->set_coeff_modulus(
          {small_mods_50_bit.begin(),
           small_mods_50_bit.begin() + sp->m_coeff_modulus.coeff_count});
    } else if (sp->m_coeff_modulus.bit_count == 60) {
      std::vector<seal::SmallModulus> small_mods_60_bit =
          seal::util::global_variables::default_small_mods_60bit;
      encryption_parms->set_coeff_modulus(
          {small_mods_60_bit.begin(),
           small_mods_60_bit.begin() + sp->m_coeff_modulus.coeff_count});
    } else {
      throw ngraph_error("Invalid coefficient bit count " +
                         to_string(sp->m_coeff_modulus.bit_count));
    }
  }

  uint64_t coeff_bit_count =
      sp->m_coeff_modulus.bit_count * sp->m_coeff_modulus.coeff_count;

  if (sp->m_security_level == 128) {
    if (custom_coeff_modulus) {
      uint64_t default_coeff_bit_count = 0;
      for (auto small_modulus :
           seal::DefaultParams::coeff_modulus_128(sp->m_poly_modulus_degree)) {
        default_coeff_bit_count += small_modulus.bit_count();
      }
      if (default_coeff_bit_count < coeff_bit_count) {
        NGRAPH_WARN << "Custom coefficient modulus has total bit count "
                    << coeff_bit_count
                    << " which is greater than the default bit count "
                    << default_coeff_bit_count
                    << ", resulting in lower security";
      }
    } else {
      encryption_parms->set_coeff_modulus(
          seal::DefaultParams::coeff_modulus_128(sp->m_poly_modulus_degree));
    }
  } else if (sp->m_security_level == 192) {
    if (custom_coeff_modulus) {
      uint64_t default_coeff_bit_count = 0;
      for (auto small_modulus :
           seal::DefaultParams::coeff_modulus_192(sp->m_poly_modulus_degree)) {
        default_coeff_bit_count += small_modulus.bit_count();
      }
      if (default_coeff_bit_count < coeff_bit_count) {
        NGRAPH_WARN << "Custom coefficient modulus has total bit count "
                    << coeff_bit_count
                    << " which is greater than the default bit count "
                    << default_coeff_bit_count
                    << ", resulting in lower security";
      }
    } else {
      encryption_parms->set_coeff_modulus(
          seal::DefaultParams::coeff_modulus_192(sp->m_poly_modulus_degree));
    }
  } else if (sp->m_security_level == 256) {
    if (custom_coeff_modulus) {
      uint64_t default_coeff_bit_count = 0;
      for (auto small_modulus :
           seal::DefaultParams::coeff_modulus_256(sp->m_poly_modulus_degree)) {
        default_coeff_bit_count += small_modulus.bit_count();
      }
      if (default_coeff_bit_count < coeff_bit_count) {
        NGRAPH_WARN << "Custom coefficient modulus has total bit count "
                    << coeff_bit_count
                    << " which is greater than the default bit count "
                    << default_coeff_bit_count
                    << ", resulting in lower security";
      }
    } else {
      encryption_parms->set_coeff_modulus(
          seal::DefaultParams::coeff_modulus_256(sp->m_poly_modulus_degree));
    }
  } else {
    throw ngraph_error("sp.security_level must be 128, 192, or 256");
  }
  return seal::SEALContext::Create(*encryption_parms);
}

namespace {
static class HESealCKKSStaticInit {
 public:
  HESealCKKSStaticInit() {
    runtime::BackendManager::register_backend("HE_SEAL_CKKS", new_ckks_backend);
  }
  ~HESealCKKSStaticInit() {}
} s_he_seal_ckks_static_init;
}  // namespace

void runtime::he::he_seal::HESealCKKSBackend::assert_valid_seal_ckks_parameter(
    const shared_ptr<runtime::he::he_seal::HESealParameter>& sp) const {
  assert_valid_seal_parameter(sp);
  if (sp->m_scheme_name != "HE_SEAL_CKKS") {
    throw ngraph_error("Invalid scheme name");
  }
}

shared_ptr<runtime::Tensor>
runtime::he::he_seal::HESealCKKSBackend::create_batched_cipher_tensor(
    const element::Type& element_type, const Shape& shape) {
  NGRAPH_INFO << "Creating batched cipher tensor with shape " << join(shape);
  auto rc = make_shared<runtime::he::HECipherTensor>(
      element_type, shape, this, create_empty_ciphertext(), true);
  return static_pointer_cast<runtime::Tensor>(rc);
}

shared_ptr<runtime::Tensor>
runtime::he::he_seal::HESealCKKSBackend::create_batched_plain_tensor(
    const element::Type& element_type, const Shape& shape) {
  NGRAPH_INFO << "Creating batched plain tensor with shape " << join(shape);
  auto rc = make_shared<runtime::he::HEPlainTensor>(
      element_type, shape, this, create_empty_plaintext(), true);
  return static_pointer_cast<runtime::Tensor>(rc);
}

void runtime::he::he_seal::HESealCKKSBackend::encode(
    shared_ptr<runtime::he::HEPlaintext>& output, const void* input,
    const element::Type& element_type, size_t count) const {
  const string type_name = element_type.c_type_string();
  if (type_name == "float") {
    if (count == 1) {
      double value = (double)(*(float*)input);
      if (m_plaintext_map.find(value) != m_plaintext_map.end()) {
        auto plain_value = static_pointer_cast<
            const runtime::he::he_seal::SealPlaintextWrapper>(
            get_valued_plaintext(value));
        output = make_shared<runtime::he::he_seal::SealPlaintextWrapper>(
            *plain_value);
      } else {
        m_ckks_encoder->encode(
            value, m_scale,
            dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(
                output)
                ->m_plaintext);
      }
    } else {
      vector<float> values{(float*)input, (float*)input + count};
      vector<double> double_values(values.begin(), values.end());

      m_ckks_encoder->encode(
          double_values, m_scale,
          dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(
              output)
              ->m_plaintext);
    }
  } else {
    NGRAPH_INFO << "Unsupported element type in encode " << type_name;
    throw ngraph_error("Unsupported element type " + type_name);
  }
}

void runtime::he::he_seal::HESealCKKSBackend::decode(
    void* output, const runtime::he::HEPlaintext* input,
    const element::Type& element_type, size_t count) const {
  const string type_name = element_type.c_type_string();

  if (count == 0) {
    throw ngraph_error("Decode called on 0 elements");
  }

  if (type_name == "float") {
    auto seal_input = dynamic_cast<const SealPlaintextWrapper*>(input);
    if (!seal_input) {
      throw ngraph_error(
          "HESealCKKSBackend::decode input is not seal plaintext");
    }
    vector<double> xs;
    m_ckks_encoder->decode(seal_input->m_plaintext, xs);
    vector<float> xs_float(xs.begin(), xs.end());

    memcpy(output, &xs_float[0], element_type.size() * count);
  } else {
    throw ngraph_error("Unsupported element type " + type_name);
  }
}

runtime::he::TCPMessage runtime::he::he_seal::HESealCKKSBackend::handle_message(
    const runtime::he::TCPMessage& message) {
  // NGRAPH_INFO << "Handling TCP Message";

  MessageType msg_type = message.message_type();

  NGRAPH_INFO << "Server got " << message_type_to_string(msg_type)
              << " message";

  if (msg_type == MessageType::execute) {
    // Get Ciphertexts from message
    std::size_t count = message.count();
    std::cout << "Got " << count << " ciphertexts " << std::endl;
    std::cout << "data size " << message.data_size() / count << std::endl;
    std::cout << "message body size " << message.body_length() << std::endl;
    std::vector<seal::Ciphertext> ciphertexts;
    size_t ciphertext_size = message.element_size();
    std::cout << "ciphertext_size" << ciphertext_size << std::endl;

    for (size_t i = 0; i < count; ++i) {
      stringstream stream;
      stream.write(message.data_ptr() + i * ciphertext_size, ciphertext_size);
      seal::Ciphertext c;

      c.load(m_context, stream);
      std::cout << "Loaded " << i << "'th ciphertext" << std::endl;
      ciphertexts.emplace_back(c);
    }
    std::vector<std::shared_ptr<runtime::he::HECiphertext>> he_cipher_inputs;

    for (const auto cipher : ciphertexts) {
      auto wrapper =
          make_shared<runtime::he::he_seal::SealCiphertextWrapper>(cipher);
      he_cipher_inputs.emplace_back(wrapper);
    }

    // Load function with parameters
    auto function = m_function_map.begin()->first;
    const ParameterVector& input_parameters = function->get_parameters();

    for (auto input_param : input_parameters) {
      std::cout << "Parameter shape " << join(input_param->get_shape(), "x")
                << std::endl;
    }
    // only support parameter size 1 for now
    assert(input_parameters.size() == 1);

    auto element_type = input_parameters[0]->get_element_type();
    bool batched = false;
    auto input_tensor = create_cipher_tensor(
        element_type, input_parameters[0]->get_shape(), batched);

    dynamic_pointer_cast<runtime::he::HECipherTensor>(input_tensor)
        ->set_elements(he_cipher_inputs);

    std::vector<shared_ptr<runtime::Tensor>> inputs{input_tensor};
    std::vector<shared_ptr<runtime::Tensor>> outputs;

    for (size_t i = 0; i < function->get_output_size(); i++) {
      auto output_type = function->get_output_element_type(i);
      auto out_shape = function->get_output_shape(i);
      auto tensor = create_cipher_tensor(output_type, out_shape, batched);
      outputs.emplace_back(tensor);
    }

    // Call function
    std::cout << "Calling function " << std::endl;
    call(function, outputs, inputs);
    size_t output_size = outputs[0]->get_element_count();
    NGRAPH_INFO << "output size " << output_size;

    // Save outputs to stringstream
    std::vector<seal::Ciphertext> seal_outputs;
    // std::vector<std::stringstream> cipher_streams(0output_size);

    std::stringstream cipher_stream;
    for (const auto& output : outputs) {
      for (const auto& element :
           dynamic_pointer_cast<runtime::he::HECipherTensor>(output)
               ->get_elements()) {
        auto wrapper =
            dynamic_pointer_cast<runtime::he::he_seal::SealCiphertextWrapper>(
                element);

        seal::Ciphertext c = wrapper->m_ciphertext;
        seal_outputs.emplace_back(c);

        c.save(cipher_stream);
      }
    }
    const std::string& cipher_str = cipher_stream.str();
    const char* cipher_cstr = cipher_str.c_str();
    std::cout << "Cipher size " << cipher_str.size() << std::endl;

    auto return_message = TCPMessage(MessageType::result, output_size,
                                     cipher_str.size(), cipher_cstr);

    return return_message;
  } else if (msg_type == MessageType::parameter_shape_request) {
    auto function = m_function_map.begin()->first;
    const ParameterVector& input_parameters = function->get_parameters();

    assert(input_parameters.size() ==
           1);  // Only support single parameter for now

    auto shape = input_parameters[0]->get_shape();

    NGRAPH_INFO << "Returning parameter shape: " << join(shape, "x");
    ;
    return TCPMessage(MessageType::parameter_shape, shape.size(),
                      sizeof(size_t) * shape.size(), (char*)shape.data());

  }

  else if (msg_type == MessageType::public_key_request) {
    seal::PublicKey pk = *m_public_key;
    stringstream stream;
    pk.save(stream);

    const std::string& pk_str = stream.str();
    const char* pk_cstr = pk_str.c_str();

    auto return_message = runtime::he::TCPMessage(MessageType::public_key, 1,
                                                  pk_str.size(), pk_cstr);

    NGRAPH_INFO << "Sending PK message back";
    return return_message;

  } else if (msg_type == MessageType::public_key) {
    size_t pk_size = message.data_size();
    std::stringstream pk_stream;
    pk_stream.write(message.data_ptr(), pk_size);
    m_public_key->load(m_context, pk_stream);

    NGRAPH_INFO << "Server loaded public key";

    auto return_message = runtime::he::TCPMessage(MessageType::public_key_ack);
    return return_message;
  } else if (msg_type == MessageType::none) {
    NGRAPH_INFO << "Server replying with none mssage";
    return message;
  } else {
    NGRAPH_INFO << "Unsupported message type in server:  "
                << message_type_to_string(msg_type);
    throw ngraph_error("Unknown message type in server");
  }
}
