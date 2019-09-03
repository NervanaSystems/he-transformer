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

#include "he_seal_cipher_tensor.hpp"
#include "ngraph/log.hpp"
#include "nlohmann/json.hpp"
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
                                       const std::vector<double>& inputs,
                                       bool complex_packing)
    : m_batch_size{batch_size},
      m_is_done(false),
      m_inputs{inputs},
      m_complex_packing(complex_packing) {
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
                                       const std::vector<float>& inputs,
                                       bool complex_packing)
    : HESealClient(hostname, port, batch_size,
                   std::vector<double>(inputs.begin(), inputs.end()),
                   complex_packing) {}

void ngraph::he::HESealClient::set_seal_context() {
  m_context = seal::SEALContext::Create(m_encryption_params, true,
                                        seal::sec_level_type::none);

  print_seal_context(*m_context);

  m_keygen = std::make_shared<seal::KeyGenerator>(m_context);
  m_relin_keys = std::make_shared<seal::RelinKeys>(m_keygen->relin_keys());
  m_public_key = std::make_shared<seal::PublicKey>(m_keygen->public_key());
  m_secret_key = std::make_shared<seal::SecretKey>(m_keygen->secret_key());
  m_encryptor = std::make_shared<seal::Encryptor>(m_context, *m_public_key);
  m_decryptor = std::make_shared<seal::Decryptor>(m_context, *m_secret_key);

  // Evaluator
  m_evaluator = std::make_shared<seal::Evaluator>(m_context);

  // Encoder
  m_ckks_encoder = std::make_shared<seal::CKKSEncoder>(m_context);

  // TODO: pick better scale?
  m_scale = ngraph::he::choose_scale(m_encryption_params.coeff_modulus());
  NGRAPH_INFO << "Client scale " << m_scale;
}

void ngraph::he::HESealClient::send_public_and_relin_keys() {
  NGRAPH_INFO << "SEnding public and relin keys";

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

  NGRAPH_INFO << "Sending pk / evk";
  write_message(proto_msg);
}

void ngraph::he::HESealClient::handle_encryption_parameters_response(
    const he_proto::TCPMessage& proto_msg) {
  NGRAPH_INFO << "Got enc parms request";
  NGRAPH_CHECK(proto_msg.has_encryption_parameters(),
               "proto_msg does not have encryption_parameters");

  const std::string& enc_parms_str =
      proto_msg.encryption_parameters().encryption_parameters();
  std::stringstream param_stream(enc_parms_str);
  m_encryption_params = seal::EncryptionParameters::Load(param_stream);

  NGRAPH_INFO << "Loaded enc parms";

  set_seal_context();
  send_public_and_relin_keys();
}

void ngraph::he::HESealClient::handle_inference_request(
    const he_proto::TCPMessage& proto_msg) {
  NGRAPH_INFO << "handle_inference_request";
  NGRAPH_CHECK(proto_msg.has_function(), "Proto msg doesn't have funtion");

  const std::string& inference_shape = proto_msg.function().function();
  json js = json::parse(inference_shape);
  std::vector<size_t> shape_dims = js.at("shape");
  ngraph::Shape shape{shape_dims};

  NGRAPH_INFO << join(shape, "x");

  size_t parameter_size =
      ngraph::shape_size(ngraph::he::HETensor::pack_shape(shape));

  NGRAPH_INFO << "Parameter size " << parameter_size;
  NGRAPH_INFO << "Client batch size " << m_batch_size;
  NGRAPH_INFO << "m_inputs.size() " << m_inputs.size();
  if (complex_packing()) {
    NGRAPH_INFO << "Client complex packing";
  }

  if (m_inputs.size() > parameter_size * m_batch_size) {
    NGRAPH_INFO << "m_inputs.size() " << m_inputs.size()
                << " > paramter_size ( " << parameter_size
                << ") * m_batch_size (" << m_batch_size << ")";
  }

  std::vector<std::shared_ptr<SealCiphertextWrapper>> ciphers(parameter_size);
  for (size_t data_idx = 0; data_idx < ciphers.size(); ++data_idx) {
    ciphers[data_idx] = std::make_shared<SealCiphertextWrapper>();
  }

  // TODO: add element type to function message
  size_t num_bytes = parameter_size * sizeof(double) * m_batch_size;
  ngraph::he::HESealCipherTensor::write(
      ciphers, m_inputs.data(), num_bytes, m_batch_size, element::f64,
      m_context->first_parms_id(), m_scale, *m_ckks_encoder, *m_encryptor,
      complex_packing());

  NGRAPH_INFO << "Saving to proto";

  const size_t maximum_message_cnt = 100;
  for (size_t parm_idx = 0; parm_idx < parameter_size;
       parm_idx += maximum_message_cnt) {
    size_t end_idx = parm_idx + maximum_message_cnt;
    if (end_idx > parameter_size) {
      end_idx = parameter_size;
    }
    if (parm_idx == end_idx) {
      break;
    }
    NGRAPH_INFO << "Creating execute message from " << parm_idx << " to "
                << end_idx;

    he_proto::TCPMessage encrypted_inputs_msg;
    encrypted_inputs_msg.set_type(he_proto::TCPMessage_Type_REQUEST);
    ngraph::he::save_to_proto(ciphers.begin() + parm_idx,
                              ciphers.begin() + end_idx, encrypted_inputs_msg);
    NGRAPH_INFO << "Writing message";
    write_message(encrypted_inputs_msg);
  }
}

void ngraph::he::HESealClient::handle_result(
    const he_proto::TCPMessage& proto_msg) {
  size_t result_count = proto_msg.ciphers_size();
  NGRAPH_INFO << "handling result count " << result_count;
  m_results.resize(result_count * m_batch_size);
  NGRAPH_INFO << "m_results size " << m_results.size();
  std::vector<std::shared_ptr<SealCiphertextWrapper>> result_ciphers(
      result_count);
#pragma omp parallel for
  for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
    ngraph::he::SealCiphertextWrapper::load(
        result_ciphers[result_idx], proto_msg.ciphers(result_idx), m_context);
  }

  size_t num_bytes = result_count * sizeof(double) * m_batch_size;
  ngraph::he::HESealCipherTensor::read(
      m_results.data(), result_ciphers, num_bytes, m_batch_size, element::f64,
      m_context->first_parms_id(), m_scale, *m_ckks_encoder, *m_decryptor,
      complex_packing());

  NGRAPH_INFO << "done handling result";
  for (const auto& elem : m_results) {
    NGRAPH_INFO << elem;
  }
  NGRAPH_INFO << "Done print resl";

  close_connection();
}

void ngraph::he::HESealClient::handle_relu_request(
    he_proto::TCPMessage&& proto_msg) {
  NGRAPH_CHECK(proto_msg.has_function(), "Proto message doesn't have function");

  proto_msg.set_type(he_proto::TCPMessage_Type_RESPONSE);

  size_t result_count = proto_msg.ciphers_size();

#pragma omp parallel for
  for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
    NGRAPH_CHECK(!proto_msg.ciphers(result_idx).known_value(),
                 "Client should not receive known-valued relu values");

    auto post_relu_cipher = std::make_shared<SealCiphertextWrapper>();
    ngraph::he::SealCiphertextWrapper::load(
        post_relu_cipher, proto_msg.ciphers(result_idx), m_context);

    ngraph::he::scalar_relu_seal(*post_relu_cipher, post_relu_cipher,
                                 m_context->first_parms_id(), m_scale,
                                 *m_ckks_encoder, *m_encryptor, *m_decryptor);
    post_relu_cipher->save(*proto_msg.mutable_ciphers(result_idx));
  }

  ngraph::he::TCPMessage relu_result_msg(proto_msg);
  write_message(relu_result_msg);
  return;
}

void ngraph::he::HESealClient::handle_bounded_relu_request(
    he_proto::TCPMessage&& proto_msg) {
  NGRAPH_CHECK(proto_msg.has_function(), "Proto message doesn't have function");

  const std::string& function = proto_msg.function().function();
  json js = json::parse(function);
  double bound = js.at("bound");

  proto_msg.set_type(he_proto::TCPMessage_Type_RESPONSE);

  size_t result_count = proto_msg.ciphers_size();

#pragma omp parallel for
  for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
    NGRAPH_CHECK(!proto_msg.ciphers(result_idx).known_value(),
                 "Client should not receive known-valued bounded relu values");

    auto post_bounded_relu_cipher = std::make_shared<SealCiphertextWrapper>();
    ngraph::he::SealCiphertextWrapper::load(
        post_bounded_relu_cipher, proto_msg.ciphers(result_idx), m_context);

    ngraph::he::scalar_bounded_relu_seal(
        *post_bounded_relu_cipher, post_bounded_relu_cipher, bound,
        m_context->first_parms_id(), m_scale, *m_ckks_encoder, *m_encryptor,
        *m_decryptor);
    post_bounded_relu_cipher->save(*proto_msg.mutable_ciphers(result_idx));
  }

  ngraph::he::TCPMessage bounded_relu_result_msg(proto_msg);
  NGRAPH_INFO << "Writing bounded relu result";
  write_message(bounded_relu_result_msg);
  return;
}

void ngraph::he::HESealClient::handle_max_pool_request(
    const he_proto::TCPMessage& proto_msg) {
  NGRAPH_INFO << "Handling max_pool request";
  NGRAPH_CHECK(proto_msg.has_function(), "Proto message doesn't have function");

  size_t cipher_count = proto_msg.ciphers_size();
  std::vector<std::shared_ptr<SealCiphertextWrapper>> max_pool_ciphers(
      cipher_count);
  std::vector<std::shared_ptr<SealCiphertextWrapper>> post_max_pool_ciphers(1);
  post_max_pool_ciphers[0] = std::make_shared<SealCiphertextWrapper>();

#pragma omp parallel for
  for (size_t cipher_idx = 0; cipher_idx < cipher_count; ++cipher_idx) {
    ngraph::he::SealCiphertextWrapper::load(
        max_pool_ciphers[cipher_idx], proto_msg.ciphers(cipher_idx), m_context);
  }

  // We currently just support max_pool with single output
  ngraph::he::max_pool_seal(
      max_pool_ciphers, post_max_pool_ciphers, Shape{1, 1, cipher_count},
      Shape{1, 1, 1}, Shape{cipher_count}, ngraph::Strides{1}, Shape{0},
      Shape{0}, m_context->first_parms_id(), m_scale, *m_ckks_encoder,
      *m_encryptor, *m_decryptor, complex_packing());

  // Create max_pool result message
  he_proto::TCPMessage proto_max_pool;
  proto_max_pool.set_type(he_proto::TCPMessage_Type_RESPONSE);
  *proto_max_pool.mutable_function() = proto_msg.function();

  post_max_pool_ciphers[0]->save(*proto_max_pool.add_ciphers());
  ngraph::he::TCPMessage max_pool_result_msg(proto_max_pool);

  NGRAPH_INFO << "Writing max_pool result";
  write_message(max_pool_result_msg);
  return;
}

void ngraph::he::HESealClient::handle_message(
    const ngraph::he::TCPMessage& message) {
  // TODO: try overwriting message?

  std::shared_ptr<he_proto::TCPMessage> proto_msg = message.proto_message();

  switch (proto_msg->type()) {
    case he_proto::TCPMessage_Type_RESPONSE: {
      NGRAPH_INFO << "Client got message RESPONSE";
      if (proto_msg->has_encryption_parameters()) {
        handle_encryption_parameters_response(*proto_msg);
      } else if (proto_msg->ciphers_size() > 0) {
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
          handle_max_pool_request(*proto_msg);
        } else {
          NGRAPH_INFO << "Unknown name " << name;
        }
      } else {
        NGRAPH_CHECK(false, "Unknown REQUEST type");
      }
      break;
    }
    case he_proto::TCPMessage_Type_UNKNOWN:
    default:
      NGRAPH_CHECK(false, "Unknonwn TCPMesage type");
  }
}

void ngraph::he::HESealClient::close_connection() {
  NGRAPH_INFO << "Closing connection";
  m_tcp_client->close();
  m_is_done = true;
  NGRAPH_INFO << "Setting is done true";
}
