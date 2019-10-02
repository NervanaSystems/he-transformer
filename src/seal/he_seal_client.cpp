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
#include <boost/asio.hpp>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "he_plain_tensor.hpp"
#include "logging/ngraph_he_log.hpp"
#include "ngraph/log.hpp"
#include "nlohmann/json.hpp"
#include "seal/he_seal_cipher_tensor.hpp"
#include "seal/he_seal_client.hpp"
#include "seal/kernel/bounded_relu_seal.hpp"
#include "seal/kernel/max_pool_seal.hpp"
#include "seal/kernel/relu_seal.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"
#include "tcp/tcp_client.hpp"
#include "tcp/tcp_message.hpp"

using json = nlohmann::json;

ngraph::he::HESealClient::HESealClient(const std::string& hostname,
                                       const size_t port,
                                       const size_t batch_size,
                                       const HETensorConfigMap<double>& inputs)
    : m_batch_size{batch_size}, m_is_done{false}, m_input_config{inputs} {
  NGRAPH_HE_LOG(5) << "Creating HESealClient";
  NGRAPH_CHECK(m_input_config.size() == 1,
               "Client supports only one input parameter");

  for (const auto& elem : inputs) {
    NGRAPH_HE_LOG(1) << "Client input tensor: " << elem.first;
  }

  boost::asio::io_context io_context;
  tcp::resolver resolver(io_context);
  auto endpoints = resolver.resolve(hostname, std::to_string(port));
  auto client_callback = [this](const ngraph::he::TCPMessage& message) {
    return handle_message(message);
  };
  m_tcp_client = std::make_unique<ngraph::he::TCPClient>(io_context, endpoints,
                                                         client_callback);
  io_context.run();
}

ngraph::he::HESealClient::HESealClient(const std::string& hostname,
                                       const size_t port,
                                       const size_t batch_size,
                                       const HETensorConfigMap<float>& inputs)
    : HESealClient(hostname, port, batch_size,
                   ngraph::he::map_to_double_map<float>(inputs)) {}

ngraph::he::HESealClient::HESealClient(const std::string& hostname,
                                       const size_t port,
                                       const size_t batch_size,
                                       const HETensorConfigMap<int64_t>& inputs)
    : HESealClient(hostname, port, batch_size,
                   ngraph::he::map_to_double_map<int64_t>(inputs)) {}

void ngraph::he::HESealClient::set_seal_context() {
  NGRAPH_HE_LOG(5) << "Client setting seal context";
  auto security_level = m_encryption_params.security_level();
  auto seal_security_level = ngraph::he::seal_security_level(security_level);

  m_context = seal::SEALContext::Create(
      m_encryption_params.seal_encryption_parameters(), true,
      seal_security_level);

  print_encryption_parameters(m_encryption_params, *m_context);

  m_keygen = std::make_shared<seal::KeyGenerator>(m_context);
  m_relin_keys = std::make_shared<seal::RelinKeys>(m_keygen->relin_keys());
  m_public_key = std::make_shared<seal::PublicKey>(m_keygen->public_key());
  m_secret_key = std::make_shared<seal::SecretKey>(m_keygen->secret_key());
  m_encryptor = std::make_shared<seal::Encryptor>(m_context, *m_public_key);
  m_decryptor = std::make_shared<seal::Decryptor>(m_context, *m_secret_key);
  m_evaluator = std::make_shared<seal::Evaluator>(m_context);
  m_ckks_encoder = std::make_shared<seal::CKKSEncoder>(m_context);
}

void ngraph::he::HESealClient::send_public_and_relin_keys() {
  he_proto::TCPMessage proto_msg;
  proto_msg.set_type(he_proto::TCPMessage_Type_RESPONSE);

  // Set public key
  std::stringstream pk_stream;
  m_public_key->save(pk_stream);
  he_proto::PublicKey public_key;
  public_key.set_public_key(pk_stream.str());
  *proto_msg.mutable_public_key() = public_key;

  // Set relinearization keys
  std::stringstream evk_stream;
  m_relin_keys->save(evk_stream);
  he_proto::EvaluationKey eval_key;
  eval_key.set_eval_key(evk_stream.str());
  *proto_msg.mutable_eval_key() = eval_key;

  write_message(std::move(proto_msg));
}

void ngraph::he::HESealClient::handle_encryption_parameters_response(
    const he_proto::TCPMessage& proto_msg) {
  NGRAPH_CHECK(proto_msg.has_encryption_parameters(),
               "proto_msg does not have encryption_parameters");

  const std::string& enc_parms_str =
      proto_msg.encryption_parameters().encryption_parameters();
  std::stringstream param_stream(enc_parms_str);

  NGRAPH_HE_LOG(3) << "Client loading encryption parameters";
  m_encryption_params = HESealEncryptionParameters::load(param_stream);

  set_seal_context();
  send_public_and_relin_keys();
}

void ngraph::he::HESealClient::handle_inference_request(
    const he_proto::TCPMessage& proto_msg) {
  NGRAPH_HE_LOG(3) << "Client handling inference request";

  // Note: the message cipher tensors are used to store the inference shapes.
  // The actual tensors returned by the client may be cipher or plain
  NGRAPH_CHECK(proto_msg.cipher_tensors_size() > 0,
               "Proto msg doesn't have any cipher tensors");

  NGRAPH_CHECK(proto_msg.cipher_tensors_size() == 1,
               "Only support 1 encrypted parameter from client");

  auto proto_tensor = proto_msg.cipher_tensors(0);
  auto proto_name = proto_tensor.name();
  auto proto_packed = proto_tensor.packed();
  auto proto_shape = proto_tensor.shape();
  ngraph::Shape shape{proto_shape.begin(), proto_shape.end()};

  NGRAPH_HE_LOG(5) << "Inference request tensor has name " << proto_name;

  bool encrypt_tensor = true;
  auto input_proto = m_input_config.find(proto_name);
  if (input_proto == m_input_config.end()) {
    // TODO: turn into hard check once provenance nodes work
    NGRAPH_WARN << "Tensor name " << proto_name << " not found";
  } else {
    std::pair<std::string, std::vector<double>> tensor_inputs =
        input_proto->second;

    if (tensor_inputs.first == "encrypt") {
      encrypt_tensor = true;
    } else if (tensor_inputs.first == "plain") {
      encrypt_tensor = false;
    } else {
      NGRAPH_WARN << "Unknown configuration " << tensor_inputs.first;
    }
  }

  NGRAPH_HE_LOG(5) << "Client received inference request with name "
                   << proto_name << ", shape {" << join(shape, "x")
                   << "}, to be "
                   << (encrypt_tensor ? "encrypted" : "plaintext");

  size_t parameter_size =
      ngraph::shape_size(ngraph::he::HETensor::pack_shape(shape));

  NGRAPH_HE_LOG(5) << "Parameter size " << parameter_size;
  NGRAPH_HE_LOG(5) << "Client batch size " << m_batch_size;
  NGRAPH_HE_LOG(5) << "m_input_config.size() " << m_input_config.size();
  if (complex_packing()) {
    NGRAPH_HE_LOG(5) << "Client complex packing";
  }

  if (m_input_config.begin()->second.second.size() >
      parameter_size * m_batch_size) {
    NGRAPH_HE_LOG(5) << "m_input_config.size() " << m_input_config.size()
                     << " > paramter_size ( " << parameter_size
                     << ") * m_batch_size (" << m_batch_size << ")";
  }

  std::vector<std::shared_ptr<SealCiphertextWrapper>> ciphers(parameter_size);
  for (size_t data_idx = 0; data_idx < ciphers.size(); ++data_idx) {
    ciphers[data_idx] = std::make_shared<SealCiphertextWrapper>();
  }

  NGRAPH_CHECK(m_input_config.size() == 1,
               "Client supports only input parameter");

  if (encrypt_tensor) {
    // TODO: add element type to function message
    size_t num_bytes = parameter_size * sizeof(double) * m_batch_size;
    ngraph::he::HESealCipherTensor::write(
        ciphers, m_input_config.begin()->second.second.data(), num_bytes,
        m_batch_size, element::f64, m_context->first_parms_id(), scale(),
        *m_ckks_encoder, *m_encryptor, complex_packing());

    std::vector<he_proto::SealCipherTensor> cipher_tensor_protos;
    ngraph::he::HESealCipherTensor::save_to_proto(cipher_tensor_protos, ciphers,
                                                  shape, "TODO");

    for (const auto& cipher_tensor_proto : cipher_tensor_protos) {
      he_proto::TCPMessage encrypted_inputs_msg;
      encrypted_inputs_msg.set_type(he_proto::TCPMessage_Type_REQUEST);

      *encrypted_inputs_msg.add_cipher_tensors() = cipher_tensor_proto;

      auto param_shape = encrypted_inputs_msg.cipher_tensors(0).shape();
      ngraph::Shape shape{param_shape.begin(), param_shape.end()};
      NGRAPH_HE_LOG(3) << "Client sending encrypted input with shape { "
                       << ngraph::join(param_shape, "x") << "}";

      write_message(std::move(encrypted_inputs_msg));
    }
  } else {
    size_t num_bytes = parameter_size * sizeof(double) * m_batch_size;
    ngraph::he::HEPlainTensor plain_tensor(element::f64, shape, proto_packed,
                                           proto_name);

    plain_tensor.write(m_input_config.begin()->second.second.data(), num_bytes);

    std::vector<he_proto::PlainTensor> plain_tensor_protos;
    plain_tensor.save_to_proto(plain_tensor_protos, "TODO");

    for (const auto& plain_tensor_proto : plain_tensor_protos) {
      he_proto::TCPMessage inputs_msg;
      inputs_msg.set_type(he_proto::TCPMessage_Type_REQUEST);

      *inputs_msg.add_plain_tensors() = plain_tensor_proto;

      auto param_shape = inputs_msg.plain_tensors(0).shape();
      ngraph::Shape shape{param_shape.begin(), param_shape.end()};
      NGRAPH_HE_LOG(3) << "Client sending plaintext input with shape "
                       << ngraph::join(param_shape, "x");

      write_message(std::move(inputs_msg));
    }
  }
}

void ngraph::he::HESealClient::handle_result(
    const he_proto::TCPMessage& proto_msg) {
  NGRAPH_HE_LOG(3) << "Client handling result";

  NGRAPH_CHECK(
      proto_msg.cipher_tensors_size() > 0 || proto_msg.plain_tensors_size() > 0,
      "Client received result with no tensors");
  NGRAPH_CHECK(proto_msg.cipher_tensors_size() == 1 ||
                   proto_msg.plain_tensors_size() == 1,
               "Client supports only results with one tensor");

  bool cipher_result = (proto_msg.cipher_tensors_size() == 1);

  if (cipher_result) {
    NGRAPH_HE_LOG(5) << "Client handling cipher result";
    auto proto_tensor = proto_msg.cipher_tensors(0);
    size_t result_count = proto_tensor.ciphertexts_size();
    m_results.resize(result_count * m_batch_size);
    std::vector<std::shared_ptr<SealCiphertextWrapper>> result_ciphers(
        result_count);
#pragma omp parallel for
    for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
      ngraph::he::SealCiphertextWrapper::load(
          result_ciphers[result_idx], proto_tensor.ciphertexts(result_idx),
          m_context);
    }

    size_t num_bytes = result_count * sizeof(double) * m_batch_size;
    ngraph::he::HESealCipherTensor::read(m_results.data(), result_ciphers,
                                         num_bytes, m_batch_size, element::f64,
                                         *m_ckks_encoder, *m_decryptor);
  } else {
    NGRAPH_HE_LOG(5) << "Client handling plain result";

    auto proto_tensor = proto_msg.plain_tensors(0);
    size_t result_count = proto_tensor.plaintexts_size();
    m_results.resize(result_count * m_batch_size);

    std::vector<ngraph::he::HEPlaintext> result_plaintexts(result_count);

    // TODO: load from protos as separate function
#pragma omp parallel for
    for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
      // Load tensor plaintexts
      auto proto_plain = proto_tensor.plaintexts(result_idx);
      result_plaintexts[result_idx] =
          ngraph::he::HEPlaintext(std::vector<double>{
              proto_plain.value().begin(), proto_plain.value().end()});

      NGRAPH_HE_LOG(5) << "Loaded plaintext " << result_plaintexts[result_idx];
    }
    ngraph::Shape shape = ngraph::he::proto_shape_to_ngraph_shape(proto_tensor);

    HEPlainTensor plain_tensor(element::f64, shape, proto_tensor.packed());
    plain_tensor.set_elements(result_plaintexts);

    size_t num_bytes = result_count * sizeof(double) * m_batch_size;
    plain_tensor.read(m_results.data(), num_bytes);
  }

  close_connection();
}

void ngraph::he::HESealClient::handle_relu_request(
    he_proto::TCPMessage&& proto_msg) {
  NGRAPH_HE_LOG(3) << "Client handling relu request";

  NGRAPH_CHECK(proto_msg.has_function(), "Proto message doesn't have function");
  NGRAPH_CHECK(proto_msg.cipher_tensors_size() > 0,
               "Client received result with no cipher tensors");
  NGRAPH_CHECK(proto_msg.cipher_tensors_size() == 1,
               "Client supports only relu requests with one cipher tensor");

  proto_msg.set_type(he_proto::TCPMessage_Type_RESPONSE);

  he_proto::SealCipherTensor* proto_tensor =
      proto_msg.mutable_cipher_tensors(0);
  size_t result_count = proto_tensor->ciphertexts_size();

#pragma omp parallel for
  for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
    NGRAPH_CHECK(!proto_tensor->ciphertexts(result_idx).known_value(),
                 "Client should not receive known-valued relu values");

    auto post_relu_cipher = std::make_shared<SealCiphertextWrapper>();
    ngraph::he::SealCiphertextWrapper::load(
        post_relu_cipher, proto_tensor->ciphertexts(result_idx), m_context);

    ngraph::he::scalar_relu_seal(*post_relu_cipher, post_relu_cipher,
                                 m_context->first_parms_id(), scale(),
                                 *m_ckks_encoder, *m_encryptor, *m_decryptor);
    post_relu_cipher->save(*proto_tensor->mutable_ciphertexts(result_idx));
  }

  ngraph::he::TCPMessage relu_result_msg(std::move(proto_msg));
  write_message(std::move(relu_result_msg));
  return;
}

void ngraph::he::HESealClient::handle_bounded_relu_request(
    he_proto::TCPMessage&& proto_msg) {
  NGRAPH_HE_LOG(3) << "Client handling bounded relu request";

  NGRAPH_CHECK(proto_msg.has_function(), "Proto message doesn't have function");
  NGRAPH_CHECK(proto_msg.cipher_tensors_size() > 0,
               "Client received result with no cipher tensors");
  NGRAPH_CHECK(
      proto_msg.cipher_tensors_size() == 1,
      "Client supports only bounded relu requests with one cipher tensor");

  const std::string& function = proto_msg.function().function();
  json js = json::parse(function);
  double bound = js.at("bound");

  proto_msg.set_type(he_proto::TCPMessage_Type_RESPONSE);

  he_proto::SealCipherTensor* proto_tensor =
      proto_msg.mutable_cipher_tensors(0);
  size_t result_count = proto_tensor->ciphertexts_size();

#pragma omp parallel for
  for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
    NGRAPH_CHECK(!proto_tensor->ciphertexts(result_idx).known_value(),
                 "Client should not receive known-valued relu values");

    auto post_relu_cipher = std::make_shared<SealCiphertextWrapper>();
    ngraph::he::SealCiphertextWrapper::load(
        post_relu_cipher, proto_tensor->ciphertexts(result_idx), m_context);

    ngraph::he::scalar_bounded_relu_seal(
        *post_relu_cipher, post_relu_cipher, bound, m_context->first_parms_id(),
        scale(), *m_ckks_encoder, *m_encryptor, *m_decryptor);
    post_relu_cipher->save(*proto_tensor->mutable_ciphertexts(result_idx));
  }

  ngraph::he::TCPMessage bounded_relu_result_msg(std::move(proto_msg));
  write_message(std::move(bounded_relu_result_msg));
  return;
}

void ngraph::he::HESealClient::handle_max_pool_request(
    he_proto::TCPMessage&& proto_msg) {
  NGRAPH_HE_LOG(3) << "Client handling maxpool request";

  NGRAPH_CHECK(proto_msg.has_function(), "Proto message doesn't have function");
  NGRAPH_CHECK(proto_msg.cipher_tensors_size() > 0,
               "Client received result with no cipher tensors");
  NGRAPH_CHECK(proto_msg.cipher_tensors_size() == 1,
               "Client supports only max pool requests with one cipher tensor");

  he_proto::SealCipherTensor* proto_tensor =
      proto_msg.mutable_cipher_tensors(0);
  size_t cipher_count = proto_tensor->ciphertexts_size();

  std::vector<std::shared_ptr<SealCiphertextWrapper>> max_pool_ciphers(
      cipher_count);
  std::vector<std::shared_ptr<SealCiphertextWrapper>> post_max_pool_ciphers(1);
  post_max_pool_ciphers[0] = std::make_shared<SealCiphertextWrapper>();

#pragma omp parallel for
  for (size_t cipher_idx = 0; cipher_idx < cipher_count; ++cipher_idx) {
    ngraph::he::SealCiphertextWrapper::load(
        max_pool_ciphers[cipher_idx], proto_tensor->ciphertexts(cipher_idx),
        m_context);
  }

  // We currently just support max_pool with single output
  ngraph::he::max_pool_seal(
      max_pool_ciphers, post_max_pool_ciphers, Shape{1, 1, cipher_count},
      Shape{1, 1, 1}, Shape{cipher_count}, ngraph::Strides{1}, Shape{0},
      Shape{0}, m_context->first_parms_id(), scale(), *m_ckks_encoder,
      *m_encryptor, *m_decryptor, complex_packing());

  proto_msg.set_type(he_proto::TCPMessage_Type_RESPONSE);
  proto_tensor->clear_ciphertexts();

  post_max_pool_ciphers[0]->save(*proto_tensor->add_ciphertexts());
  ngraph::he::TCPMessage max_pool_result_msg(std::move(proto_msg));
  write_message(std::move(max_pool_result_msg));
  return;
}

void ngraph::he::HESealClient::handle_message(
    const ngraph::he::TCPMessage& message) {
  // TODO: try overwriting message?

  NGRAPH_HE_LOG(3) << "Client handling message";

  std::shared_ptr<he_proto::TCPMessage> proto_msg = message.proto_message();

  switch (proto_msg->type()) {
    case he_proto::TCPMessage_Type_RESPONSE: {
      if (proto_msg->has_encryption_parameters()) {
        handle_encryption_parameters_response(*proto_msg);
      } else if (proto_msg->cipher_tensors_size() > 0 ||
                 proto_msg->plain_tensors_size() > 0) {
        handle_result(*proto_msg);
      } else {
        NGRAPH_CHECK(false, "Unknown RESPONSE type");
      }
      break;
    }
    case he_proto::TCPMessage_Type_REQUEST: {
      if (proto_msg->has_function()) {
        const std::string& function = proto_msg->function().function();
        json js = json::parse(function);
        auto name = js.at("function");

        if (name == "Parameter") {
          handle_inference_request(*proto_msg);
        } else if (name == "Relu") {
          handle_relu_request(std::move(*proto_msg));
        } else if (name == "BoundedRelu") {
          handle_bounded_relu_request(std::move(*proto_msg));
        } else if (name == "MaxPool") {
          handle_max_pool_request(std::move(*proto_msg));
        } else {
          NGRAPH_HE_LOG(5) << "Unknown name " << name;
        }
      } else {
        NGRAPH_CHECK(false, "Unknown REQUEST type");
      }
      break;
    }
    case he_proto::TCPMessage_Type_UNKNOWN:
    default:
      NGRAPH_CHECK(false, "Unknonwn TCPMessage type");
  }
}

void ngraph::he::HESealClient::close_connection() {
  NGRAPH_HE_LOG(5) << "Closing connection";
  m_tcp_client->close();
  m_is_done = true;
}
