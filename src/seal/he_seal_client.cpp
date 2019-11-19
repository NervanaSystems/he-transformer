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

#include "seal/he_seal_client.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "boost/asio.hpp"
#include "logging/ngraph_he_log.hpp"
#include "ngraph/log.hpp"
#include "nlohmann/json.hpp"
#include "seal/kernel/bounded_relu_seal.hpp"
#include "seal/kernel/max_pool_seal.hpp"
#include "seal/kernel/relu_seal.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"
#include "tcp/tcp_client.hpp"
#include "tcp/tcp_message.hpp"

using json = nlohmann::json;

namespace ngraph::runtime::he {

HESealClient::HESealClient(const std::string& hostname, const size_t port,
                           const size_t batch_size,
                           const HETensorConfigMap<double>& inputs)
    : m_batch_size{batch_size}, m_input_config{inputs} {
  NGRAPH_HE_LOG(5) << "Creating HESealClient from config";
  NGRAPH_CHECK(m_input_config.size() == 1,
               "Client supports only one input parameter");

  for (const auto& elem : inputs) {
    NGRAPH_HE_LOG(1) << "Client input tensor: " << elem.first;
  }

  boost::asio::io_context io_context;
  boost::asio::ip::tcp::resolver resolver(io_context);
  auto endpoints = resolver.resolve(hostname, std::to_string(port));
  auto client_callback = [this](const TCPMessage& message) {
    return handle_message(message);
  };
  m_tcp_client =
      std::make_unique<TCPClient>(io_context, endpoints, client_callback);
  io_context.run();
}

HESealClient::HESealClient(const std::string& hostname, const size_t port,
                           const size_t batch_size,
                           const HETensorConfigMap<float>& inputs)
    : HESealClient(hostname, port, batch_size,
                   map_to_double_map<float>(inputs)) {}

HESealClient::HESealClient(const std::string& hostname, const size_t port,
                           const size_t batch_size,
                           const HETensorConfigMap<int64_t>& inputs)
    : HESealClient(hostname, port, batch_size,
                   map_to_double_map<int64_t>(inputs)) {}

void HESealClient::set_seal_context() {
  NGRAPH_HE_LOG(5) << "Client setting seal context";
  auto seal_sec_level =
      seal_security_level(m_encryption_params.security_level());

  m_context = seal::SEALContext::Create(
      m_encryption_params.seal_encryption_parameters(), true, seal_sec_level);

  print_encryption_parameters(m_encryption_params, *m_context);

  m_keygen = std::make_shared<seal::KeyGenerator>(m_context);
  if (m_context->using_keyswitching()) {
    m_relin_keys = std::make_shared<seal::RelinKeys>(m_keygen->relin_keys());
  }
  m_public_key = std::make_shared<seal::PublicKey>(m_keygen->public_key());
  m_secret_key = std::make_shared<seal::SecretKey>(m_keygen->secret_key());
  m_encryptor = std::make_shared<seal::Encryptor>(m_context, *m_public_key);
  m_decryptor = std::make_shared<seal::Decryptor>(m_context, *m_secret_key);
  m_evaluator = std::make_shared<seal::Evaluator>(m_context);
  m_ckks_encoder = std::make_shared<seal::CKKSEncoder>(m_context);
}

void HESealClient::init_aby_executor() {
  NGRAPH_INFO << "Initializing ABY executor";
  if (m_aby_executor == nullptr) {
    m_aby_executor = std::make_unique<aby::ABYClientExecutor>(
        std::string("yao"), *this, m_hostname);
  }
}

void HESealClient::send_public_and_relin_keys() {
  NGRAPH_HE_LOG(3) << "Client sending public and relin keys";
  pb::TCPMessage message;
  message.set_type(pb::TCPMessage_Type_RESPONSE);

  // Set public key
  std::stringstream pk_stream;
  m_public_key->save(pk_stream);
  pb::PublicKey public_key;
  public_key.set_public_key(pk_stream.str());
  *message.mutable_public_key() = public_key;

  // Set relinearization keys
  if (m_context->using_keyswitching()) {
    std::stringstream evk_stream;
    m_relin_keys->save(evk_stream);
    pb::EvaluationKey eval_key;
    eval_key.set_eval_key(evk_stream.str());
    *message.mutable_eval_key() = eval_key;
  }

  write_message(TCPMessage(std::move(message)));
}

void HESealClient::handle_encryption_parameters_response(
    const pb::TCPMessage& message) {
  NGRAPH_HE_LOG(3) << "Client handling encryption parameters message";

  NGRAPH_CHECK(message.has_encryption_parameters(),
               "message does not have encryption_parameters");

  const std::string& enc_parms_str =
      message.encryption_parameters().encryption_parameters();
  std::stringstream param_stream(enc_parms_str);

  NGRAPH_HE_LOG(3) << "Client loading encryption parameters from stream size "
                   << enc_parms_str.size();
  m_encryption_params = HESealEncryptionParameters::load(param_stream);

  set_seal_context();
  send_public_and_relin_keys();
}

void HESealClient::handle_inference_request(const pb::TCPMessage& message) {
  NGRAPH_HE_LOG(3) << "Client handling inference request";

  // Note: the message tensors are used to store the inference shapes.
  NGRAPH_CHECK(message.he_tensors_size() > 0,
               "Proto msg doesn't have any cipher tensors");

  NGRAPH_CHECK(message.he_tensors_size() == 1,
               "Only support 1 encrypted parameter from client");

  NGRAPH_CHECK(m_input_config.size() == 1,
               "Client supports only input parameter");

  const auto& proto_tensor = message.he_tensors(0);
  auto& proto_name = proto_tensor.name();
  auto proto_shape = proto_tensor.shape();
  Shape shape{proto_shape.begin(), proto_shape.end()};

  NGRAPH_HE_LOG(5) << "Inference request tensor has name " << proto_name;

  bool encrypt_tensor = true;
  auto input_proto = m_input_config.find(proto_name);
  NGRAPH_CHECK(input_proto != m_input_config.end(), "Tensor name ", proto_name,
               " not found");

  auto& [input_config, input_data] = input_proto->second;
  static std::unordered_set<std::string> known_configs{"encrypt", "plain"};

  NGRAPH_CHECK(known_configs.find(input_config) != known_configs.end(),
               "Unknown configuration ", input_config);

  if (input_config == "encrypt") {
    encrypt_tensor = true;
  } else if (input_config == "plain") {
    encrypt_tensor = false;
  }

  NGRAPH_HE_LOG(5) << "Client received inference request with name "
                   << proto_name << ", " << shape << ", to be "
                   << (encrypt_tensor ? "encrypted" : "plaintext");

  NGRAPH_HE_LOG(5) << "Client batch size " << m_batch_size;
  NGRAPH_HE_LOG(5) << "m_input_config.size() " << m_input_config.size();
  if (complex_packing()) {
    NGRAPH_HE_LOG(5) << "Client complex packing";
  }

  size_t parameter_size = shape_size(HETensor::pack_shape(shape));
  NGRAPH_HE_LOG(5) << "Client parameter_size " << parameter_size;

  NGRAPH_CHECK(input_data.size() == parameter_size * m_batch_size,
               "incorrect input_data.size() ", input_data.size(),
               ", expected  ", parameter_size * m_batch_size,
               " (parameter_size=", parameter_size,
               "), (batch_size=", m_batch_size, ")");

  shape = HETensor::unpack_shape(shape, m_batch_size);
  auto element_type = element::f64;

  auto he_tensor = HETensor(
      element_type, shape, proto_tensor.packed(),
      m_encryption_params.complex_packing(), encrypt_tensor, *m_ckks_encoder,
      m_context, *m_encryptor, *m_decryptor, m_encryption_params, proto_name);

  size_t num_bytes = parameter_size * sizeof(double) * m_batch_size;
  NGRAPH_HE_LOG(3) << "Writing to tensor";
  he_tensor.write(input_data.data(), num_bytes);

  std::vector<pb::HETensor> tensor_protos;
  NGRAPH_HE_LOG(3) << "Writing to protos";
  he_tensor.write_to_protos(tensor_protos);
  for (const auto& tensor_proto : tensor_protos) {
    pb::TCPMessage inputs_msg;
    inputs_msg.set_type(pb::TCPMessage_Type_REQUEST);
    *inputs_msg.add_he_tensors() = tensor_proto;

    auto param_shape = inputs_msg.he_tensors(0).shape();
    NGRAPH_HE_LOG(3) << "Client sending encrypted input with shape "
                     << Shape{param_shape.begin(), param_shape.end()};
    write_message(TCPMessage(std::move(inputs_msg)));
  }
}

void HESealClient::handle_result(const pb::TCPMessage& message) {
  NGRAPH_HE_LOG(3) << "Client handling result";

  NGRAPH_CHECK(message.he_tensors_size() > 0,
               "Client received result with no tensors");
  NGRAPH_CHECK(message.he_tensors_size() == 1,
               "Client supports only results with one tensor");

  const auto& proto_tensor = message.he_tensors(0);

  if (m_result_tensor == nullptr) {
    m_result_tensor = HETensor::load_from_proto_tensor(
        proto_tensor, *m_ckks_encoder, m_context, *m_encryptor, *m_decryptor,
        m_encryption_params);
  } else {
    HETensor::load_from_proto_tensor(m_result_tensor, proto_tensor, m_context);
  }

  if (m_result_tensor->done_loading()) {
    size_t data_size =
        m_result_tensor->data().size() * m_result_tensor->get_batch_size();
    m_results.resize(data_size);

    const auto& type = m_result_tensor->get_element_type();
    size_t num_bytes = data_size * type.size();
    auto bytes = ngraph_malloc(num_bytes);
    m_result_tensor->read(bytes, num_bytes);

    for (size_t i = 0; i < data_size; ++i) {
      void* addr =
          static_cast<void*>(static_cast<char*>(bytes) + i * type.size());
      m_results[i] = type_to_double(addr, type);
    }

    ngraph_free(bytes);
    close_connection();
  }
}

void HESealClient::handle_relu_request(pb::TCPMessage&& message) {
  NGRAPH_HE_LOG(3) << "Client handling relu request";

  NGRAPH_CHECK(message.has_function(), "Proto message doesn't have function");
  NGRAPH_CHECK(message.he_tensors_size() > 0,
               "Client received result with no tensors");
  NGRAPH_CHECK(message.he_tensors_size() == 1,
               "Client supports only relu requests with one tensor");

  message.set_type(pb::TCPMessage_Type_RESPONSE);

  pb::HETensor* proto_tensor = message.mutable_he_tensors(0);
  auto he_tensor = HETensor::load_from_proto_tensor(
      *proto_tensor, *m_ckks_encoder, m_context, *m_encryptor, *m_decryptor,
      m_encryption_params);

  const std::string& function = message.function().function();
  const json& js = json::parse(function);
  bool enable_gc = string_to_bool(std::string(js.at("enable_gc")));
  NGRAPH_INFO << "Client relu with gc? " << enable_gc;

  if (enable_gc) {
    NGRAPH_HE_LOG(3) << "Client relu with GC";

    init_aby_executor();

    m_aby_executor->prepare_aby_circuit(function, he_tensor);
    m_aby_executor->run_aby_circuit(function, he_tensor);
    NGRAPH_INFO << "Client done running aby circuit";

  } else {
    size_t result_count = proto_tensor->data_size();
#pragma omp parallel for
    for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
      scalar_relu_seal(he_tensor->data(result_idx), he_tensor->data(result_idx),
                       m_context->first_parms_id(), scale(), *m_ckks_encoder,
                       *m_encryptor, *m_decryptor, m_context);
    }
  }

  std::vector<pb::HETensor> proto_output_tensors;
  he_tensor->write_to_protos(proto_output_tensors);

  NGRAPH_CHECK(proto_output_tensors.size() == 1,
               "Only support single-output tensors");
  *proto_tensor = proto_output_tensors[0];

  write_message(TCPMessage(std::move(message)));
}

void HESealClient::handle_bounded_relu_request(pb::TCPMessage&& message) {
  NGRAPH_HE_LOG(3) << "Client handling bounded relu request";

  NGRAPH_CHECK(message.has_function(), "Proto message doesn't have function");
  NGRAPH_CHECK(message.he_tensors_size() > 0,
               "Client received result with no tensors");
  NGRAPH_CHECK(message.he_tensors_size() == 1,
               "Client supports only relu requests with one tensor");

  const std::string& function = message.function().function();
  json js = json::parse(function);
  double bound = js.at("bound");

  message.set_type(pb::TCPMessage_Type_RESPONSE);

  pb::HETensor* proto_tensor = message.mutable_he_tensors(0);
  auto he_tensor = HETensor::load_from_proto_tensor(
      *proto_tensor, *m_ckks_encoder, m_context, *m_encryptor, *m_decryptor,
      m_encryption_params);

  bool enable_gc = string_to_bool(std::string(js.at("enable_gc")));

  if (enable_gc) {
    NGRAPH_HE_LOG(3) << "Client bounded relu with GC";
    init_aby_executor();

    m_aby_executor->prepare_aby_circuit(function, he_tensor);
    m_aby_executor->run_aby_circuit(function, he_tensor);
    NGRAPH_INFO << "Client done running aby circuit";

  } else {
    size_t result_count = proto_tensor->data_size();
#pragma omp parallel for
    for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
      scalar_bounded_relu_seal(
          he_tensor->data(result_idx), he_tensor->data(result_idx), bound,
          m_context->first_parms_id(), scale(), *m_ckks_encoder, *m_encryptor,
          *m_decryptor, m_context);
    }
  }
  std::vector<pb::HETensor> proto_output_tensors;
  he_tensor->write_to_protos(proto_output_tensors);
  NGRAPH_CHECK(proto_output_tensors.size() == 1,
               "Only support single-output tensors");
  *proto_tensor = proto_output_tensors[0];

  write_message(TCPMessage(std::move(message)));
}

void HESealClient::handle_max_pool_request(pb::TCPMessage&& message) {
  NGRAPH_HE_LOG(3) << "Client handling maxpool request";

  NGRAPH_CHECK(message.has_function(), "Proto message doesn't have function ");
  NGRAPH_CHECK(message.he_tensors_size() > 0,
               " Client received result with no tensors ");
  NGRAPH_CHECK(message.he_tensors_size() == 1,
               "Client supports only max pool requests with one tensor");

  pb::HETensor* proto_tensor = message.mutable_he_tensors(0);
  size_t cipher_count = proto_tensor->data_size();

  std::vector<HEType> max_pool_ciphers(
      cipher_count, HEType(HEPlaintext(m_batch_size), false));
  std::vector<HEType> post_max_pool_ciphers(
      {HEType(HEPlaintext(m_batch_size), false)});

  auto he_tensor = HETensor::load_from_proto_tensor(
      *proto_tensor, *m_ckks_encoder, m_context, *m_encryptor, *m_decryptor,
      m_encryption_params);

  // We currently just support max_pool with single output
  auto post_max_he_tensor =
      HETensor(he_tensor->get_element_type(), Shape{m_batch_size, 1},
               he_tensor->is_packed(), complex_packing(), true, *m_ckks_encoder,
               m_context, *m_encryptor, *m_decryptor, m_encryption_params);

  max_pool_seal(he_tensor->data(), post_max_he_tensor.data(),
                Shape{1, 1, cipher_count}, Shape{1, 1, 1}, Shape{cipher_count},
                Strides{1}, Shape{0}, Shape{0}, m_context->first_parms_id(),
                scale(), *m_ckks_encoder, *m_encryptor, *m_decryptor,
                m_context);

  message.set_type(pb::TCPMessage_Type_RESPONSE);
  message.clear_he_tensors();

  std::vector<pb::HETensor> proto_output_tensors;
  post_max_he_tensor.write_to_protos(proto_output_tensors);
  NGRAPH_CHECK(proto_output_tensors.size() == 1,
               "Only support single-output tensors");

  *message.add_he_tensors() = proto_output_tensors[0];
  TCPMessage max_pool_result_msg(std::move(message));
  write_message(std::move(max_pool_result_msg));
}

void HESealClient::handle_message(const TCPMessage& message) {
  NGRAPH_HE_LOG(3) << "Client handling message";

  std::shared_ptr<pb::TCPMessage> proto_msg = message.proto_message();

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-enum"
  switch (proto_msg->type()) {
    case pb::TCPMessage_Type_RESPONSE: {
      if (proto_msg->has_encryption_parameters()) {
        handle_encryption_parameters_response(*proto_msg);
      } else if (proto_msg->he_tensors_size() > 0) {
        handle_result(*proto_msg);
      } else {
        NGRAPH_CHECK(false, "Unknown RESPONSE type");
      }
      break;
    }
    case pb::TCPMessage_Type_REQUEST: {
      NGRAPH_CHECK(proto_msg->has_function(), "Unknown request type");

      const std::string& function = proto_msg->function().function();
      json js = json::parse(function);
      auto name = js.at("function");

      // TODO(fboemer): Move to any_of in message.proto
      static std::unordered_set<std::string> s_known_names{
          "Parameter", "Relu", "BoundedRelu", "MaxPool"};

      NGRAPH_CHECK(s_known_names.find(name) != s_known_names.end(),
                   "Unknown name ", name);

      if (name == "Parameter") {
        handle_inference_request(*proto_msg);
      } else if (name == "Relu") {
        handle_relu_request(std::move(*proto_msg));
      } else if (name == "BoundedRelu") {
        handle_bounded_relu_request(std::move(*proto_msg));
      } else if (name == "MaxPool") {
        handle_max_pool_request(std::move(*proto_msg));
      }
      break;
    }
    case pb::TCPMessage_Type_UNKNOWN:
    default:
      NGRAPH_CHECK(false, "Unknown TCPMessage type");
  }
#pragma clang diagnostic pop
}

std::vector<double> HESealClient::get_results() {
  NGRAPH_INFO << "Client waiting for results";

  std::unique_lock<std::mutex> mlock(m_is_done_mutex);
  m_is_done_cond.wait(mlock, [this]() { return this->is_done(); });
  return m_results;
}

void HESealClient::close_connection() {
  NGRAPH_HE_LOG(5) << "Closing connection";
  m_tcp_client->close();

  std::lock_guard<std::mutex> guard(m_is_done_mutex);
  m_is_done = true;
  m_is_done_cond.notify_all();
}

}  // namespace ngraph::runtime::he
