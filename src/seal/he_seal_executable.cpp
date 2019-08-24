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

#include <functional>
#include <limits>
#include <unordered_set>

#include "he_plain_tensor.hpp"
#include "he_seal_cipher_tensor.hpp"
#include "he_tensor.hpp"
#include "kernel/add_seal.hpp"
#include "kernel/avg_pool_seal.hpp"
#include "kernel/batch_norm_inference_seal.hpp"
#include "kernel/bounded_relu_seal.hpp"
#include "kernel/broadcast_seal.hpp"
#include "kernel/concat_seal.hpp"
#include "kernel/constant_seal.hpp"
#include "kernel/convolution_seal.hpp"
#include "kernel/dot_seal.hpp"
#include "kernel/max_pool_seal.hpp"
#include "kernel/minimum_seal.hpp"
#include "kernel/multiply_seal.hpp"
#include "kernel/negate_seal.hpp"
#include "kernel/pad_seal.hpp"
#include "kernel/relu_seal.hpp"
#include "kernel/reshape_seal.hpp"
#include "kernel/result_seal.hpp"
#include "kernel/reverse_seal.hpp"
#include "kernel/slice_seal.hpp"
#include "kernel/subtract_seal.hpp"
#include "kernel/sum_seal.hpp"
#include "ngraph/assertion.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/passthrough.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "op/bounded_relu.hpp"
#include "pass/he_fusion.hpp"
#include "pass/he_liveness.hpp"
#include "pass/supported_ops.hpp"
#include "protos/message.pb.h"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_executable.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"

using ngraph::descriptor::layout::DenseTensorLayout;
using json = nlohmann::json;

ngraph::he::HESealExecutable::HESealExecutable(
    const std::shared_ptr<Function>& function,
    bool enable_performance_collection, HESealBackend& he_seal_backend,
    bool encrypt_data, bool encrypt_model, bool pack_data, bool complex_packing,
    bool enable_client)
    : m_he_seal_backend(he_seal_backend),
      m_encrypt_data(encrypt_data),
      m_encrypt_model(encrypt_model),
      m_pack_data(pack_data),
      m_complex_packing(complex_packing),
      m_verbose_all_ops(false),
      m_enable_client(enable_client),
      m_client_setup(false),
      m_batch_size(1),
      m_port(34000),
      m_relu_done_count(0),
      m_max_pool_done(false),
      m_session_started(false),
      m_client_inputs_received(false) {
  m_context = he_seal_backend.get_context();

  if (std::getenv("NGRAPH_VOPS") != nullptr) {
    std::string verbose_ops_str(std::getenv("NGRAPH_VOPS"));
    verbose_ops_str = ngraph::to_lower(verbose_ops_str);
    if (verbose_ops_str == "all") {
      m_verbose_all_ops = true;
    }
    std::vector<std::string> verbose_ops_vec =
        split(verbose_ops_str, ',', true);
    m_verbose_ops =
        std::set<std::string>{verbose_ops_vec.begin(), verbose_ops_vec.end()};

    if (m_verbose_ops.find("all") != m_verbose_ops.end()) {
      m_verbose_all_ops = true;
    }
  }

  m_is_compiled = true;
  ngraph::pass::Manager pass_manager;
  pass_manager.register_pass<ngraph::pass::LikeReplacement>();
  pass_manager.register_pass<ngraph::pass::AssignLayout<DenseTensorLayout>>();
  pass_manager.register_pass<ngraph::pass::CoreFusion>();
  if (std::getenv("STOP_CONST_FOLD") == nullptr) {
    pass_manager.register_pass<ngraph::pass::ConstantFolding>();
  }
  pass_manager.run_passes(function);
  ngraph::pass::Manager pass_manager_he;
  pass_manager_he.register_pass<ngraph::he::pass::HEFusion>();
  pass_manager_he.register_pass<ngraph::he::pass::HELiveness>();
  pass_manager_he.register_pass<ngraph::he::pass::SupportedOps>(
      [this](const ngraph::Node& op) {
        return m_he_seal_backend.is_supported(op);
      });
  pass_manager_he.run_passes(function);

  for (const std::shared_ptr<Node>& node : function->get_ordered_ops()) {
    m_wrapped_nodes.emplace_back(node);
  }
  set_parameters_and_results(*function);

  // Constant, for example, cannot be packed
  if (m_pack_data) {
    if (get_parameters().size() > 0) {
      const Shape& shape = (get_parameters()[0])->get_shape();
      NGRAPH_CHECK(shape.size() > 0, "Parameter shape empty");

      m_batch_size = shape[0];
      for (auto& parameter : get_parameters()) {
        const Shape& param_shape = parameter->get_shape();
        NGRAPH_CHECK(param_shape.size() > 0, "Parameter shape empty");
        size_t new_batch_size = param_shape[0];
        NGRAPH_CHECK(
            new_batch_size == m_batch_size, "Function contains ",
            get_parameters().size(),
            " parameters, which do not all imply the same batch size.");
      }

      size_t max_batch_size =
          m_he_seal_backend.get_ckks_encoder()->slot_count();
      if (m_complex_packing) {
        max_batch_size *= 2;
      }
      NGRAPH_CHECK(m_batch_size <= max_batch_size, "Batch size ", m_batch_size,
                   " too large (maximum ", max_batch_size, ")");
    }
  }

  if (m_enable_client) {
    NGRAPH_INFO << "Setting up client in constructor";
    client_setup();
  }
}

void ngraph::he::HESealExecutable::check_client_supports_function() {
  NGRAPH_CHECK(get_parameters().size() == 1,
               "HESealExecutable only supports parameter size 1 (got ",
               get_parameters().size(), ")");

  // only support function output size 1 for now
  NGRAPH_CHECK(get_results().size() == 1,
               "HESealExecutable only supports output size 1 (got ",
               get_results().size(), "");
}

void ngraph::he::HESealExecutable::client_setup() {
  if (!m_client_setup) {
    NGRAPH_INFO << "Enable client";
    check_client_supports_function();

    // Start server
    NGRAPH_INFO << "Starting server";
    start_server();

    NGRAPH_INFO << "Creatign new parms message";

    std::stringstream param_stream;
    m_he_seal_backend.get_encryption_parameters().save(param_stream);

    he_proto::EncryptionParameters proto_parms;
    *proto_parms.mutable_encryption_parameters() = param_stream.str();

    he_proto::TCPMessage proto_msg;
    *proto_msg.mutable_encryption_parameters() = proto_parms;
    proto_msg.set_type(he_proto::TCPMessage_Type_RESPONSE);

    ngraph::he::TCPMessage parms_message(proto_msg);
    NGRAPH_INFO << "Created PB parms message";
    std::unique_lock<std::mutex> mlock(m_session_mutex);
    m_session_cond.wait(mlock,
                        std::bind(&HESealExecutable::session_started, this));
    m_session->write_message(std::move(parms_message));

    m_client_setup = true;
  } else {
    NGRAPH_INFO << "Client already setup";
  }
}

void ngraph::he::HESealExecutable::accept_connection() {
  NGRAPH_INFO << "Server accepting connections";
  auto server_callback = bind(&ngraph::he::HESealExecutable::handle_message,
                              this, std::placeholders::_1);

  m_acceptor->async_accept([this, server_callback](boost::system::error_code ec,
                                                   tcp::socket socket) {
    if (!ec) {
      NGRAPH_INFO << "Connection accepted";
      m_session =
          std::make_shared<TCPSession>(std::move(socket), server_callback);
      m_session->start();
      NGRAPH_INFO << "Session started";

      std::lock_guard<std::mutex> guard(m_session_mutex);
      m_session_started = true;
      m_session_cond.notify_one();
    } else {
      NGRAPH_INFO << "error accepting connection " << ec.message();
      // accept_connection();
    }
  });
}

void ngraph::he::HESealExecutable::start_server() {
  tcp::resolver resolver(m_io_context);
  tcp::endpoint server_endpoints(tcp::v4(), m_port);
  m_acceptor = std::make_unique<tcp::acceptor>(m_io_context, server_endpoints);
  boost::asio::socket_base::reuse_address option(true);
  m_acceptor->set_option(option);

  accept_connection();
  m_thread = std::thread([this]() { m_io_context.run(); });
}

void ngraph::he::HESealExecutable::load_public_key(
    const he_proto::TCPMessage& proto_msg) {
  NGRAPH_INFO << "Loading public key";
  NGRAPH_CHECK(proto_msg.has_public_key(), "proto_msg doesn't have public key");

  seal::PublicKey key;
  const std::string& pk_str = proto_msg.public_key().public_key();
  std::stringstream key_stream(pk_str);
  key.load(m_context, key_stream);
  m_he_seal_backend.set_public_key(key);

  NGRAPH_INFO << "Server set public key";

  m_client_public_key_set = true;
}

void ngraph::he::HESealExecutable::load_eval_key(
    const he_proto::TCPMessage& proto_msg) {
  NGRAPH_CHECK(proto_msg.has_eval_key(), "proto_msg doesn't have eval key");
  NGRAPH_INFO << "Loading eval key";

  seal::RelinKeys keys;
  const std::string& evk_str = proto_msg.eval_key().eval_key();
  std::stringstream key_stream(evk_str);
  keys.load(m_context, key_stream);
  m_he_seal_backend.set_relin_keys(keys);

  NGRAPH_INFO << "Server set eval key";

  m_client_eval_key_set = true;
}

void ngraph::he::HESealExecutable::send_inference_shape() {
  NGRAPH_INFO << "Sending inference shape";
  m_sent_inference_shape = true;

  const ParameterVector& input_parameters = get_parameters();

  // TODO: support > 1 input parameter
  NGRAPH_CHECK(input_parameters.size() == 1,
               "Only support input parameters size 1");
  json js;
  auto& param = input_parameters[0];
  js["shape"] = param->get_shape();
  js["function"] = "Parameter";

  he_proto::TCPMessage proto_msg;
  proto_msg.set_type(he_proto::TCPMessage_Type_REQUEST);

  he_proto::Function f;
  f.set_function(js.dump());
  *proto_msg.mutable_function() = f;

  NGRAPH_INFO << "Sending inference shape " << js.dump();

  ngraph::he::TCPMessage execute_msg(proto_msg);
  m_session->write_message(std::move(execute_msg));
}

void ngraph::he::HESealExecutable::handle_relu_result(
    const he_proto::TCPMessage& proto_msg) {
  std::lock_guard<std::mutex> guard(m_relu_mutex);
  size_t message_count = proto_msg.ciphers_size();

#pragma omp parallel for
  for (size_t element_idx = 0; element_idx < message_count; ++element_idx) {
    std::shared_ptr<ngraph::he::SealCiphertextWrapper> new_cipher;
    ngraph::he::SealCiphertextWrapper::load(
        new_cipher, proto_msg.ciphers(element_idx), m_context);

    m_relu_ciphertexts[m_unknown_relu_idx[element_idx + m_relu_done_count]] =
        new_cipher;
  }
  m_relu_done_count += message_count;
  m_relu_cond.notify_all();
}
void ngraph::he::HESealExecutable::handle_bounded_relu_result(
    const he_proto::TCPMessage& proto_msg) {
  handle_relu_result(proto_msg);
}

void ngraph::he::HESealExecutable::handle_max_pool_result(
    const he_proto::TCPMessage& proto_msg) {
  std::lock_guard<std::mutex> guard(m_max_pool_mutex);
  size_t message_count = proto_msg.ciphers_size();
  NGRAPH_INFO << "handle_max_pool_result with count " << message_count;

  NGRAPH_CHECK(message_count == 1,
               "Maxpool only supports message count 1, got ", message_count);

  std::shared_ptr<ngraph::he::SealCiphertextWrapper> new_cipher;
  ngraph::he::SealCiphertextWrapper::load(new_cipher, proto_msg.ciphers(0),
                                          m_context);

  m_max_pool_ciphertexts.emplace_back(new_cipher);
  m_max_pool_done = true;
  m_max_pool_cond.notify_all();
}

void ngraph::he::HESealExecutable::handle_message(
    const ngraph::he::TCPMessage& message) {
  std::shared_ptr<he_proto::TCPMessage> proto_msg = message.proto_message();

  switch (proto_msg->type()) {
    case he_proto::TCPMessage_Type_RESPONSE: {
      NGRAPH_INFO << "Server got new message RESPONSE";
      if (proto_msg->has_public_key()) {
        load_public_key(*proto_msg);
      }
      if (proto_msg->has_eval_key()) {
        load_eval_key(*proto_msg);
      }
      if (!m_sent_inference_shape && m_client_public_key_set &&
          m_client_eval_key_set) {
        send_inference_shape();
      }

      if (proto_msg->has_function()) {
        const std::string& function = proto_msg->function().function();
        json js = json::parse(function);

        auto name = js.at("function");
        if (name == "Relu") {
          handle_relu_result(*proto_msg);
        } else if (name == "BoundedRelu") {
          handle_bounded_relu_result(*proto_msg);
        } else if (name == "MaxPool") {
          handle_max_pool_result(*proto_msg);
        } else {
          NGRAPH_INFO << "Unknown name " << name;
        }
      }
      break;
    }
    case he_proto::TCPMessage_Type_REQUEST: {
      NGRAPH_INFO << "Server got new message REQUEST";
      if (proto_msg->ciphers_size() > 0) {
        NGRAPH_INFO << "Got input ciphers";
        handle_client_ciphers(*proto_msg);
      }
      break;
    }
    case he_proto::TCPMessage_Type_UNKNOWN:
    default:
      NGRAPH_CHECK(false, "Unknonwn TCPMesage type");
  }
}

void ngraph::he::HESealExecutable::handle_client_ciphers(
    const he_proto::TCPMessage& proto_msg) {
  size_t count = proto_msg.ciphers_size();
  NGRAPH_INFO << "Loading " << count << " ciphertexts";

  std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>
      he_cipher_inputs(count);
#pragma omp parallel for
  for (size_t cipher_idx = 0; cipher_idx < count; ++cipher_idx) {
    ngraph::he::SealCiphertextWrapper::load(
        he_cipher_inputs[cipher_idx], proto_msg.ciphers(cipher_idx), m_context);
  }

  // only support parameter size 1 for now
  NGRAPH_CHECK(get_parameters().size() == 1,
               "HESealExecutable only supports parameter size 1 (got ",
               get_parameters().size(), ")");

  // only support function output size 1 for now
  NGRAPH_CHECK(get_results().size() == 1,
               "HESealExecutable only supports output size 1 (got ",
               get_results().size(), "");

  // Load function with parameters
  size_t num_param_elements = 0;
  const ParameterVector& input_parameters = get_parameters();
  for (auto input_param : input_parameters) {
    NGRAPH_INFO << "param shape " << join(input_param->get_shape(), "x");
    num_param_elements += shape_size(input_param->get_shape());
  }

  num_param_elements /= m_batch_size;
  NGRAPH_CHECK(count == num_param_elements, "Count ", count,
               " does not match number of parameter elements ( ",
               num_param_elements, ")");

  NGRAPH_INFO << "Setting m_client_inputs";
  size_t parameter_size_index = 0;
  for (auto input_param : input_parameters) {
    const auto& shape = input_param->get_shape();
    size_t param_size = shape_size(shape) / m_batch_size;
    auto element_type = input_param->get_element_type();
    auto input_tensor =
        std::dynamic_pointer_cast<ngraph::he::HESealCipherTensor>(
            m_he_seal_backend.create_cipher_tensor(
                element_type, input_param->get_shape(), m_pack_data,
                "client_parameter"));

    std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>
        cipher_elements{
            he_cipher_inputs.begin() + parameter_size_index,
            he_cipher_inputs.begin() + parameter_size_index + param_size};

    NGRAPH_CHECK(cipher_elements.size() == param_size,
                 "Incorrect number of elements for parameter");

    input_tensor->set_elements(cipher_elements);
    for (auto& cipher_elem : cipher_elements) {
      cipher_elem->complex_packing() = m_complex_packing;
    }
    m_client_inputs.emplace_back(input_tensor);
    parameter_size_index += param_size;
  }

  NGRAPH_CHECK(m_client_inputs.size() == get_parameters().size(),
               "Client inputs size ", m_client_inputs.size(), "; expected ",
               get_parameters().size());

  NGRAPH_INFO << "Notifiyng client inputs received";

  std::lock_guard<std::mutex> guard(m_client_inputs_mutex);
  m_client_inputs_received = true;
  m_client_inputs_cond.notify_all();
}

std::vector<ngraph::runtime::PerformanceCounter>
ngraph::he::HESealExecutable::get_performance_data() const {
  std::vector<runtime::PerformanceCounter> rc;
  for (const std::pair<std::shared_ptr<const Node>, stopwatch> p :
       m_timer_map) {
    rc.emplace_back(p.first, p.second.get_total_microseconds(),
                    p.second.get_call_count());
  }
  return rc;
}

bool ngraph::he::HESealExecutable::call(
    const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& server_inputs) {
  validate(outputs, server_inputs);

  if (m_encrypt_data) {
    NGRAPH_INFO << "Encrypting data";
  }
  if (m_pack_data) {
    NGRAPH_INFO << "Batching data with batch size " << m_batch_size;
  }
  if (m_encrypt_model) {
    NGRAPH_INFO << "Encrypting model";
  }
  if (m_complex_packing) {
    NGRAPH_INFO << "Complex packing";
  }

  if (m_enable_client) {
    NGRAPH_INFO << "Waiting until m_client_inputs.size() == "
                << server_inputs.size();

    std::unique_lock<std::mutex> mlock(m_client_inputs_mutex);
    m_client_inputs_cond.wait(
        mlock, std::bind(&HESealExecutable::client_inputs_received, this));
    NGRAPH_INFO << "client_inputs_received";

    NGRAPH_CHECK(m_client_inputs.size() == server_inputs.size(),
                 "Recieved incorrect number of inputs from client (got ",
                 m_client_inputs.size(), ", expectd ", server_inputs.size());
  }

  // convert inputs to HETensor
  std::vector<std::shared_ptr<ngraph::he::HETensor>> he_inputs;
  if (m_enable_client) {
    NGRAPH_DEBUG << "Processing client inputs";
    for (auto& tv : m_client_inputs) {
      he_inputs.push_back(std::static_pointer_cast<ngraph::he::HETensor>(tv));
    }
  } else {
    NGRAPH_DEBUG << "Processing server inputs";
    for (auto& tv : server_inputs) {
      auto he_input = std::dynamic_pointer_cast<ngraph::he::HETensor>(tv);
      NGRAPH_CHECK(he_input != nullptr, "server input is not he tensor");
      he_inputs.push_back(he_input);
    }
  }

  // convert outputs to HETensor
  std::vector<std::shared_ptr<ngraph::he::HETensor>> he_outputs;
  for (auto& tv : outputs) {
    he_outputs.push_back(std::static_pointer_cast<ngraph::he::HETensor>(tv));
  }

  std::unordered_map<ngraph::descriptor::Tensor*,
                     std::shared_ptr<ngraph::he::HETensor>>
      tensor_map;

  // map function params -> HETensor
  size_t input_count = 0;
  for (auto param : get_parameters()) {
    for (size_t param_idx = 0; param_idx < param->get_output_size();
         ++param_idx) {
      descriptor::Tensor* tv = param->get_output_tensor_ptr(param_idx).get();

      if (!m_enable_client && m_encrypt_data) {
        NGRAPH_DEBUG << "Encrypting parameter " << param_idx;
        auto plain_input = std::dynamic_pointer_cast<ngraph::he::HEPlainTensor>(
            he_inputs[input_count]);

        NGRAPH_CHECK(plain_input != nullptr, "Input is not plain tensor");
        std::string name = tv->get_name();

        auto cipher_input = std::dynamic_pointer_cast<HESealCipherTensor>(
            m_he_seal_backend.create_cipher_tensor(
                plain_input->get_element_type(), plain_input->get_shape(),
                m_pack_data, name));

#pragma omp parallel for
        for (size_t plain_idx = 0;
             plain_idx < plain_input->get_batched_element_count();
             ++plain_idx) {
          encrypt(cipher_input->get_element(plain_idx),
                  plain_input->get_element(plain_idx),
                  m_he_seal_backend.get_context()->first_parms_id(),
                  plain_input->get_element_type(),
                  m_he_seal_backend.get_scale(),
                  *m_he_seal_backend.get_ckks_encoder(),
                  *m_he_seal_backend.get_encryptor(), m_complex_packing);
        }
        NGRAPH_DEBUG << "Done encrypting parameter";
        plain_input->reset();
        tensor_map.insert({tv, cipher_input});
        input_count++;
      } else {
        tensor_map.insert({tv, he_inputs[input_count++]});
      }
    }
  }

  // map function outputs -> HostTensor
  for (size_t output_count = 0; output_count < get_results().size();
       ++output_count) {
    auto output = get_results()[output_count];
    if (!std::dynamic_pointer_cast<op::Result>(output)) {
      throw ngraph_error("One of function's outputs isn't op::Result");
    }
    ngraph::descriptor::Tensor* tv = output->get_output_tensor_ptr(0).get();
    tensor_map.insert({tv, he_outputs[output_count++]});
  }

  // for each ordered op in the graph
  for (const NodeWrapper& wrapped : m_wrapped_nodes) {
    auto op = wrapped.get_node();
    auto type_id = wrapped.get_typeid();
    bool verbose = verbose_op(*op);

    if (verbose) {
      NGRAPH_INFO << "\033[1;32m"
                  << "[ " << op->get_name() << " ]"
                  << "\033[0m";
      if (type_id == OP_TYPEID::Constant) {
        NGRAPH_INFO << "Constant shape {" << join(op->get_shape()) << "}";
      }
    }

    if (type_id == OP_TYPEID::Parameter) {
      if (verbose) {
        NGRAPH_INFO << "Parameter shape {" << join(op->get_shape()) << "}";
      }
      continue;
    }
    m_timer_map[op].start();

    // get op inputs from map
    std::vector<std::shared_ptr<ngraph::he::HETensor>> op_inputs;
    for (auto input : op->inputs()) {
      descriptor::Tensor* tensor = &input.get_tensor();
      op_inputs.push_back(tensor_map.at(tensor));
    }

    if (m_enable_client && type_id == OP_TYPEID::Result) {
      // Client outputs remain ciphertexts, so don't perform result op on them
      NGRAPH_INFO << "Setting client outputs";
      m_client_outputs = op_inputs;
    }

    // get op outputs from map or create
    std::vector<std::shared_ptr<ngraph::he::HETensor>> op_outputs;
    for (size_t i = 0; i < op->get_output_size(); ++i) {
      auto tensor = &op->output(i).get_tensor();
      auto it = tensor_map.find(tensor);
      if (it == tensor_map.end()) {
        // The output tensor is not in the tensor map so create a new tensor
        const Shape& shape = op->get_output_shape(i);
        const element::Type& element_type = op->get_output_element_type(i);
        std::string name = op->output(i).get_tensor().get_name();

        // Plaintext output only if all inputs are plaintext
        bool plain_out = all_of(
            op_inputs.begin(), op_inputs.end(),
            [](std::shared_ptr<ngraph::he::HETensor> op_input) {
              return std::dynamic_pointer_cast<HEPlainTensor>(op_input) !=
                     nullptr;
            });
        if (op->is_constant()) {
          plain_out = !m_encrypt_model;
        }
        bool packed_out =
            std::any_of(op_inputs.begin(), op_inputs.end(),
                        [](std::shared_ptr<ngraph::he::HETensor> he_tensor) {
                          return he_tensor->is_packed();
                        });
        // Avoid broadcasting from constant to output with batch size first
        // dimension This happens because not every constant is packed, for
        // examples convolution kernels.
        if (m_pack_data && shape.size() > 0 && shape[0] == m_batch_size &&
            op->description() == "Broadcast") {
          packed_out = true;
        }

        if (plain_out) {
          auto out_tensor = std::make_shared<ngraph::he::HEPlainTensor>(
              element_type, shape, m_he_seal_backend, packed_out, name);
          tensor_map.insert({tensor, out_tensor});
        } else {
          auto out_tensor = std::make_shared<ngraph::he::HESealCipherTensor>(
              element_type, shape, m_he_seal_backend, packed_out, name);
          tensor_map.insert({tensor, out_tensor});
        }
      }
      op_outputs.push_back(tensor_map.at(tensor));
    }

    // get op type
    element::Type base_type;
    if (op->get_inputs().empty()) {
      base_type = op->get_element_type();
    } else {
      base_type = op->get_inputs().at(0).get_tensor().get_element_type();
    }

    generate_calls(base_type, wrapped, op_outputs, op_inputs);
    m_timer_map[op].stop();

    // delete any obsolete tensors
    for (const descriptor::Tensor* t : op->liveness_free_list) {
      bool erased = false;
      for (auto it = tensor_map.begin(); it != tensor_map.end(); ++it) {
        const std::string& it_name = it->second->get_name();
        if (it_name == t->get_name()) {
          tensor_map.erase(it);
          erased = true;
          break;
        }
      }
      if (!erased) {
        NGRAPH_DEBUG << "Failed to erase " << t->get_name()
                     << " from tensor map";
      }
    }
    if (verbose) {
      NGRAPH_INFO << "\033[1;31m" << op->get_name() << " took "
                  << m_timer_map[op].get_milliseconds() << "ms"
                  << "\033[0m";
    }
  }
  size_t total_time = 0;
  for (const auto& elem : m_timer_map) {
    total_time += elem.second.get_milliseconds();
  }
  if (verbose_op("total")) {
    NGRAPH_INFO << "\033[1;32m"
                << "Total time " << total_time << " (ms) \033[0m";
  }

  // Send outputs to client.
  if (m_enable_client) {
    send_client_results();
  }
  return true;
}

void ngraph::he::HESealExecutable::send_client_results() {
  NGRAPH_INFO << "Sending outputs to client";
  NGRAPH_CHECK(m_client_outputs.size() == 1,
               "HESealExecutable only supports output size 1 (got ",
               get_results().size(), "");

  he_proto::TCPMessage proto_msg;
  proto_msg.set_type(he_proto::TCPMessage_Type_RESPONSE);

  std::vector<seal::Ciphertext> seal_output;

  auto output_cipher_tensor =
      std::dynamic_pointer_cast<HESealCipherTensor>(m_client_outputs[0]);
  NGRAPH_CHECK(output_cipher_tensor != nullptr,
               "Client outputs are not HESealCipherTensor");

  for (const auto& ciphertext_wrapper : output_cipher_tensor->get_elements()) {
    ciphertext_wrapper->save(*proto_msg.add_ciphers());
  }

  NGRAPH_INFO << "Writing Result message with " << proto_msg.ciphers_size()
              << " ciphertexts ";

  ngraph::he::TCPMessage result_msg(proto_msg);

  m_session->write_message(std::move(result_msg));

  std::unique_lock<std::mutex> mlock(m_result_mutex);

  // Wait until message is written
  std::condition_variable& writing_cond = m_session->is_writing_cond();
  writing_cond.wait(mlock, [this] { return !m_session->is_writing(); });
}

void ngraph::he::HESealExecutable::generate_calls(
    const element::Type& type, const NodeWrapper& node_wrapper,
    const std::vector<std::shared_ptr<HETensor>>& out,
    const std::vector<std::shared_ptr<HETensor>>& args) {
  const Node& node = *node_wrapper.get_node();
  bool verbose = verbose_op(node);
  std::string node_op = node.description();
  std::shared_ptr<HESealCipherTensor> arg0_cipher = nullptr;
  std::shared_ptr<HEPlainTensor> arg0_plain = nullptr;
  std::shared_ptr<HESealCipherTensor> arg1_cipher = nullptr;
  std::shared_ptr<HEPlainTensor> arg1_plain = nullptr;
  auto out0_cipher = std::dynamic_pointer_cast<HESealCipherTensor>(out[0]);
  auto out0_plain = std::dynamic_pointer_cast<HEPlainTensor>(out[0]);

  // TODO: move to static function
  auto lazy_rescaling = [this](auto& cipher_tensor,
                               bool verbose_rescaling = true) {
    if (m_he_seal_backend.naive_rescaling()) {
      return;
    }
    if (verbose_rescaling) {
      NGRAPH_INFO << "Rescaling " << cipher_tensor->num_ciphertexts()
                  << " ciphertexts";
    }

    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();
    size_t new_chain_index = std::numeric_limits<size_t>::max();

    bool all_known_values = true;
    for (size_t cipher_idx = 0; cipher_idx < cipher_tensor->num_ciphertexts();
         ++cipher_idx) {
      auto& cipher = cipher_tensor->get_element(cipher_idx);
      if (!cipher->known_value()) {
        size_t curr_chain_index =
            get_chain_index(cipher->ciphertext(), m_he_seal_backend);
        if (curr_chain_index == 0) {
          new_chain_index = 0;
        } else {
          new_chain_index = curr_chain_index - 1;
        }
        all_known_values = false;
        break;
      }
    }

    if (all_known_values) {
      if (verbose_rescaling) {
        NGRAPH_INFO << "Skipping rescaling because all values are known";
      }
      return;
    }

    NGRAPH_CHECK(new_chain_index != std::numeric_limits<size_t>::max(),
                 "Lazy rescaling called on cipher tensor of all known values");
    if (new_chain_index == 0) {
      if (verbose_rescaling) {
        NGRAPH_INFO << "Skipping rescaling to chain index 0";
      }
      return;
    }
    if (verbose_rescaling) {
      NGRAPH_INFO << "New chain index " << new_chain_index;
    }

#pragma omp parallel for
    for (size_t i = 0; i < cipher_tensor->num_ciphertexts(); ++i) {
      auto cipher = cipher_tensor->get_element(i);
      if (!cipher->known_value()) {
        m_he_seal_backend.get_evaluator()->rescale_to_next_inplace(
            cipher->ciphertext());
      }
    }
    if (verbose_rescaling) {
      auto t2 = Clock::now();
      NGRAPH_INFO << "Rescale_xxx took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
                                                                           t1)
                         .count()
                  << "ms";
    }
  };

  std::vector<Shape> packed_arg_shapes{};
  std::vector<Shape> unpacked_arg_shapes{};
  for (size_t arg_idx = 0; arg_idx < args.size(); ++arg_idx) {
    Shape arg_shape = node.get_input_shape(arg_idx);
    unpacked_arg_shapes.emplace_back(arg_shape);
    if (m_pack_data) {
      arg_shape = ngraph::he::HETensor::pack_shape(arg_shape);
    }
    packed_arg_shapes.emplace_back(arg_shape);
  }

  Shape out_shape{};
  Shape packed_out_shape{};
  if (node.get_output_size() > 0) {
    NGRAPH_CHECK(node.get_output_size() == 1,
                 "Only support single-output functions");
    out_shape = node.get_output_shape(0);
    packed_out_shape = out_shape;
    if (m_pack_data) {
      packed_out_shape = ngraph::he::HETensor::pack_shape(packed_out_shape);
    }
  }

  if (args.size() > 0) {
    arg0_cipher = std::dynamic_pointer_cast<HESealCipherTensor>(args[0]);
    arg0_plain = std::dynamic_pointer_cast<HEPlainTensor>(args[0]);
    NGRAPH_CHECK(arg0_cipher == nullptr || arg0_plain == nullptr,
                 "arg0 is neither cipher nor plain");
    NGRAPH_CHECK(!(arg0_cipher != nullptr && arg0_plain != nullptr),
                 "arg0 is both cipher and plain?");
  }
  if (args.size() > 1) {
    arg1_cipher = std::dynamic_pointer_cast<HESealCipherTensor>(args[1]);
    arg1_plain = std::dynamic_pointer_cast<HEPlainTensor>(args[1]);
    NGRAPH_CHECK(arg1_cipher == nullptr || arg1_plain == nullptr,
                 "arg1 is neither cipher nor plain");
    NGRAPH_CHECK(!(arg1_cipher != nullptr && arg1_plain != nullptr),
                 "arg1 is both cipher and plain?");
  }

  if (verbose) {
    std::stringstream ss;
    ss << "Inputs: ";
    if (arg0_cipher != nullptr) {
      ss << "Cipher";
    } else if (arg0_plain != nullptr) {
      ss << "Plain";
    }
    if (arg1_cipher != nullptr) {
      ss << ", Cipher";
    } else if (arg1_plain != nullptr) {
      ss << ", Plain";
    }
    for (size_t arg_ind = 2; arg_ind < args.size(); ++arg_ind) {
      auto arg = args[arg_ind];
      if (std::dynamic_pointer_cast<HESealCipherTensor>(arg) != nullptr) {
        ss << ", Cipher";
      } else if (std::dynamic_pointer_cast<HEPlainTensor>(arg) != nullptr) {
        ss << ", Plain";
      } else {
        throw ngraph_error("argument is neither plain nor cipher tensor");
      }
    }

    NGRAPH_INFO << ss.str();
    ss.str("");
    ss << "Outputs: ";
    if (out0_cipher != nullptr) {
      ss << "Cipher";
    } else if (out0_plain != nullptr) {
      ss << "Plain";
    }
    NGRAPH_INFO << ss.str();
  }

// We want to check that every OP_TYPEID enumeration is included in the list.
// These GCC flags enable compile-time checking so that if an enumeration
// is not in the list an error is generated.
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
  switch (node_wrapper.get_typeid()) {
    case OP_TYPEID::Add: {
      if (arg0_cipher != nullptr && arg1_cipher != nullptr &&
          out0_cipher != nullptr) {
        ngraph::he::add_seal(
            arg0_cipher->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), type, m_he_seal_backend,
            out0_cipher->get_batched_element_count());
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        ngraph::he::add_seal(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), type, m_he_seal_backend,
            out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && arg1_cipher != nullptr &&
                 out0_cipher != nullptr) {
        ngraph::he::add_seal(
            arg0_plain->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), type, m_he_seal_backend,
            out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        ngraph::he::add_seal(
            arg0_plain->get_elements(), arg1_plain->get_elements(),
            out0_plain->get_elements(), type, m_he_seal_backend,
            out0_plain->get_batched_element_count());
      } else {
        throw ngraph_error("Add types not supported.");
      }
      break;
    }
    case OP_TYPEID::AvgPool: {
      const op::AvgPool* avg_pool = static_cast<const op::AvgPool*>(&node);
      Shape op_in_shape = unpacked_arg_shapes[0];
      Shape op_out_shape = packed_out_shape;

      if (verbose) {
        NGRAPH_INFO << "AvgPool " << join(op_in_shape, "x") << " => "
                    << join(op_out_shape, "x");
      }

      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        ngraph::he::avg_pool_seal(
            arg0_cipher->get_elements(), out0_cipher->get_elements(),
            op_in_shape, op_out_shape, avg_pool->get_window_shape(),
            avg_pool->get_window_movement_strides(),
            avg_pool->get_padding_below(), avg_pool->get_padding_above(),
            avg_pool->get_include_padding_in_avg_computation(),
            m_he_seal_backend);
        lazy_rescaling(out0_cipher, verbose);

      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        ngraph::he::avg_pool_seal(
            arg0_plain->get_elements(), out0_plain->get_elements(), op_in_shape,
            op_out_shape, avg_pool->get_window_shape(),
            avg_pool->get_window_movement_strides(),
            avg_pool->get_padding_below(), avg_pool->get_padding_above(),
            avg_pool->get_include_padding_in_avg_computation(),
            m_he_seal_backend);

      } else {
        throw ngraph_error("AvgPool types not supported.");
      }
      break;
    }
    case OP_TYPEID::BatchNormInference: {
      const ngraph::op::BatchNormInference* bn =
          static_cast<const ngraph::op::BatchNormInference*>(&node);
      double eps = bn->get_eps_value();
      NGRAPH_CHECK(args.size() == 5, "BatchNormInference has ", args.size(),
                   "arguments (expected 5).");

      auto gamma = std::dynamic_pointer_cast<HEPlainTensor>(args[0]);
      auto beta = std::dynamic_pointer_cast<HEPlainTensor>(args[1]);
      auto input = std::dynamic_pointer_cast<HESealCipherTensor>(args[2]);
      auto mean = std::dynamic_pointer_cast<HEPlainTensor>(args[3]);
      auto variance = std::dynamic_pointer_cast<HEPlainTensor>(args[4]);

      NGRAPH_CHECK(out0_cipher != nullptr, "BatchNorm output not cipher");
      NGRAPH_CHECK(gamma != nullptr, "BatchNorm gamma not plain");
      NGRAPH_CHECK(beta != nullptr, "BatchNorm beta not plain");
      NGRAPH_CHECK(input != nullptr, "BatchNorm input not cipher");
      NGRAPH_CHECK(mean != nullptr, "BatchNorm mean not plaintext");
      NGRAPH_CHECK(variance != nullptr, "BatchNorm variance not plaintext");

      ngraph::he::batch_norm_inference_seal(
          eps, gamma->get_elements(), beta->get_elements(),
          input->get_elements(), mean->get_elements(), variance->get_elements(),
          out0_cipher->get_elements(), packed_arg_shapes[2], m_batch_size,
          m_he_seal_backend);
      break;
    }
    case OP_TYPEID::BoundedRelu: {
      const op::BoundedRelu* bounded_relu =
          static_cast<const op::BoundedRelu*>(&node);
      float alpha = bounded_relu->get_alpha();

      if (arg0_plain != nullptr && out0_plain != nullptr) {
        size_t output_size = arg0_plain->get_batched_element_count();
        NGRAPH_CHECK(output_size == arg0_plain->num_plaintexts(),
                     "output size ", output_size,
                     " doesn't match number of elements",
                     out0_plain->num_plaintexts());
        ngraph::he::bounded_relu_seal(arg0_plain->get_elements(),
                                      out0_plain->get_elements(), output_size,
                                      alpha);
        break;
      }

      if (arg0_cipher == nullptr || out0_cipher == nullptr) {
        throw ngraph_error("Relu types not supported");
      }

      if (!m_enable_client) {
        NGRAPH_WARN << "Performing BoundedRelu without client is not "
                       "privacy-preserving";
        size_t output_size = arg0_cipher->get_batched_element_count();
        NGRAPH_CHECK(output_size == arg0_cipher->num_ciphertexts(),
                     "output size ", output_size,
                     " doesn't match number of elements",
                     out0_cipher->num_ciphertexts());
        ngraph::he::bounded_relu_seal(arg0_cipher->get_elements(),
                                      out0_cipher->get_elements(), output_size,
                                      alpha, m_he_seal_backend);
        break;
      }
      NGRAPH_CHECK(alpha == 6.0f,
                   "Client supports BoundeRelu(6) only; got BoundedRelu(",
                   alpha, ")");
      handle_server_relu_op(arg0_cipher, out0_cipher, node_wrapper);
      break;
    }
    case OP_TYPEID::Broadcast: {
      const op::Broadcast* broadcast = static_cast<const op::Broadcast*>(&node);
      AxisSet broadcast_axes = broadcast->get_broadcast_axes();
      Shape in_shape = unpacked_arg_shapes[0];
      Shape broadcast_out_shape = out_shape;
      if (out_shape[0] == m_batch_size) {
        broadcast_out_shape = packed_out_shape;
      }

      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        ngraph::he::broadcast_seal(arg0_cipher->get_elements(),
                                   out0_cipher->get_elements(), in_shape,
                                   broadcast_out_shape, broadcast_axes);
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        ngraph::he::broadcast_seal(arg0_plain->get_elements(),
                                   out0_plain->get_elements(), in_shape,
                                   broadcast_out_shape, broadcast_axes);
      } else {
        throw ngraph_error("Broadcast types not supported.");
      }
      break;
    }

    case OP_TYPEID::BroadcastLike:
      break;
    case OP_TYPEID::Concat: {
      const op::Concat* concat = static_cast<const op::Concat*>(&node);

      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        std::vector<Shape> in_shapes;
        std::vector<
            std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>>
            in_args;

        for (std::shared_ptr<HETensor> arg : args) {
          std::shared_ptr<HESealCipherTensor> arg_cipher =
              std::dynamic_pointer_cast<HESealCipherTensor>(arg);
          if (arg_cipher == nullptr) {
            throw ngraph_error("Concat type not consistent");
          }
          in_args.push_back(arg_cipher->get_elements());
          in_shapes.push_back(arg_cipher->get_packed_shape());
        }
        ngraph::he::concat_seal(in_args, out0_cipher->get_elements(), in_shapes,
                                packed_out_shape,
                                concat->get_concatenation_axis());
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        std::vector<Shape> in_shapes;
        std::vector<std::vector<ngraph::he::HEPlaintext>> in_args;

        for (std::shared_ptr<HETensor> arg : args) {
          auto arg_plain = std::dynamic_pointer_cast<HEPlainTensor>(arg);
          if (arg_plain == nullptr) {
            throw ngraph_error("Concat type not consistent");
          }
          in_args.emplace_back(arg_plain->get_elements());
          in_shapes.push_back(arg_plain->get_packed_shape());
        }
        ngraph::he::concat_seal(in_args, out0_plain->get_elements(), in_shapes,
                                packed_out_shape,
                                concat->get_concatenation_axis());
      } else {
        throw ngraph_error("Concat types not supported.");
      }
      break;
    }
    case OP_TYPEID::Constant: {
      const op::Constant* constant = static_cast<const op::Constant*>(&node);

      if (out0_plain != nullptr) {
        ngraph::he::constant_seal(out0_plain->get_elements(), type,
                                  constant->get_data_ptr(), m_he_seal_backend,
                                  out0_plain->get_batched_element_count());
      } else if (out0_cipher != nullptr) {
        ngraph::he::constant_seal(out0_cipher->get_elements(), type,
                                  constant->get_data_ptr(), m_he_seal_backend,
                                  out0_cipher->get_batched_element_count());
      } else {
        throw ngraph_error("Constant type not supported.");
      }
      break;
    }
    case OP_TYPEID::Convolution: {
      const op::Convolution* c = static_cast<const op::Convolution*>(&node);
      auto window_movement_strides = c->get_window_movement_strides();
      auto window_dilation_strides = c->get_window_dilation_strides();
      auto padding_below = c->get_padding_below();
      auto padding_above = c->get_padding_above();
      auto data_dilation_strides = c->get_data_dilation_strides();

      Shape in_shape0 = packed_arg_shapes[0];
      Shape in_shape1 = unpacked_arg_shapes[1];

      if (arg0_cipher != nullptr && arg1_cipher != nullptr &&
          out0_cipher != nullptr) {
        ngraph::he::convolution_seal(
            arg0_cipher->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), in_shape0, in_shape1, packed_out_shape,
            window_movement_strides, window_dilation_strides, padding_below,
            padding_above, data_dilation_strides, 0, 1, 1, 0, 0, 1, false, type,
            m_batch_size, m_he_seal_backend, verbose);
        lazy_rescaling(out0_cipher, verbose);
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        ngraph::he::convolution_seal(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), in_shape0, in_shape1, packed_out_shape,
            window_movement_strides, window_dilation_strides, padding_below,
            padding_above, data_dilation_strides, 0, 1, 1, 0, 0, 1, false, type,
            m_batch_size, m_he_seal_backend, verbose);
        lazy_rescaling(out0_cipher, verbose);
      } else if (arg0_plain != nullptr && arg1_cipher != nullptr &&
                 out0_cipher != nullptr) {
        ngraph::he::convolution_seal(
            arg0_plain->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), in_shape0, in_shape1, packed_out_shape,
            window_movement_strides, window_dilation_strides, padding_below,
            padding_above, data_dilation_strides, 0, 1, 1, 0, 0, 1, false, type,
            m_batch_size, m_he_seal_backend, verbose);
        lazy_rescaling(out0_cipher, verbose);
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        ngraph::he::convolution_seal(
            arg0_plain->get_elements(), arg1_plain->get_elements(),
            out0_plain->get_elements(), in_shape0, in_shape1, packed_out_shape,
            window_movement_strides, window_dilation_strides, padding_below,
            padding_above, data_dilation_strides, 0, 1, 1, 0, 0, 1, false, type,
            m_batch_size, m_he_seal_backend, verbose);
      } else {
        throw ngraph_error("Convolution types not supported.");
      }
      break;
    }
    case OP_TYPEID::Dot: {
      const op::Dot* dot = static_cast<const op::Dot*>(&node);
      Shape in_shape0 = packed_arg_shapes[0];
      Shape in_shape1 = unpacked_arg_shapes[1];

      if (verbose) {
        NGRAPH_INFO << join(in_shape0, "x") << " dot " << join(in_shape1, "x");
      }
      if (arg0_cipher != nullptr && arg1_cipher != nullptr &&
          out0_cipher != nullptr) {
        ngraph::he::dot_seal(
            arg0_cipher->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), in_shape0, in_shape1, packed_out_shape,
            dot->get_reduction_axes_count(), type, m_he_seal_backend);
        lazy_rescaling(out0_cipher, verbose);
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        ngraph::he::dot_seal(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), in_shape0, in_shape1, packed_out_shape,
            dot->get_reduction_axes_count(), type, m_he_seal_backend);
        lazy_rescaling(out0_cipher, verbose);
      } else if (arg0_plain != nullptr && arg1_cipher != nullptr &&
                 out0_cipher != nullptr) {
        ngraph::he::dot_seal(
            arg0_plain->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), in_shape0, in_shape1, packed_out_shape,
            dot->get_reduction_axes_count(), type, m_he_seal_backend);
        lazy_rescaling(out0_cipher, verbose);
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        ngraph::he::dot_seal(
            arg0_plain->get_elements(), arg1_plain->get_elements(),
            out0_plain->get_elements(), in_shape0, in_shape1,
            out0_plain->get_packed_shape(), dot->get_reduction_axes_count(),
            type, m_he_seal_backend);
      } else {
        throw ngraph_error("Dot types not supported.");
      }
      break;
    }
    case OP_TYPEID::MaxPool: {
      NGRAPH_INFO << "max pool op";

      const op::MaxPool* max_pool = static_cast<const op::MaxPool*>(&node);
      if (arg0_plain != nullptr && out0_plain != nullptr) {
        ngraph::he::max_pool_seal(
            arg0_plain->get_elements(), out0_plain->get_elements(),
            unpacked_arg_shapes[0], out0_plain->get_packed_shape(),
            max_pool->get_window_shape(),
            max_pool->get_window_movement_strides(),
            max_pool->get_padding_below(), max_pool->get_padding_above());
        break;
      }
      if (arg0_cipher == nullptr || out0_cipher == nullptr) {
        throw ngraph_error("MaxPool supports only Cipher, Cipher");
      }

      if (!m_enable_client) {
        NGRAPH_WARN
            << "Performing MaxPool without client is not privacy-preserving";
        size_t output_size = arg0_cipher->get_batched_element_count();
        NGRAPH_CHECK(output_size == arg0_cipher->num_ciphertexts(),
                     "output size ", output_size,
                     " doesn't match number of elements",
                     out0_cipher->num_ciphertexts());
        ngraph::he::max_pool_seal(
            arg0_cipher->get_elements(), out0_cipher->get_elements(),
            unpacked_arg_shapes[0], out0_cipher->get_packed_shape(),
            max_pool->get_window_shape(),
            max_pool->get_window_movement_strides(),
            max_pool->get_padding_below(), max_pool->get_padding_above(),
            m_he_seal_backend);
        break;
      }

      handle_server_max_pool_op(arg0_cipher, out0_cipher, node_wrapper);
      break;
    }
    case OP_TYPEID::Minimum: {
      if (arg0_plain != nullptr && arg1_plain != nullptr &&
          out0_plain != nullptr) {
        ngraph::he::minimum_seal(arg0_plain->get_elements(),
                                 arg1_plain->get_elements(),
                                 out0_plain->get_elements(),
                                 out0_plain->get_batched_element_count());
        break;
      }
      throw ngraph_error("Minimum op unsupported for ciphertexts");
    }
    case OP_TYPEID::Multiply: {
      if (arg0_cipher != nullptr && arg1_cipher != nullptr &&
          out0_cipher != nullptr) {
        ngraph::he::multiply_seal(
            arg0_cipher->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), type, m_he_seal_backend,
            out0_cipher->get_batched_element_count());
        lazy_rescaling(out0_cipher, verbose);
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        ngraph::he::multiply_seal(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), type, m_he_seal_backend,
            out0_cipher->get_batched_element_count());
        lazy_rescaling(out0_cipher, verbose);
      } else if (arg0_plain != nullptr && arg1_cipher != nullptr &&
                 out0_cipher != nullptr) {
        ngraph::he::multiply_seal(
            arg0_plain->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), type, m_he_seal_backend,
            out0_cipher->get_batched_element_count());
        lazy_rescaling(out0_cipher, verbose);
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        ngraph::he::multiply_seal(
            arg0_plain->get_elements(), arg1_plain->get_elements(),
            out0_plain->get_elements(), type, m_he_seal_backend,
            out0_plain->get_batched_element_count());
      } else {
        throw ngraph_error("Multiply types not supported.");
      }
      break;
    }
    case OP_TYPEID::Negative: {
      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        ngraph::he::negate_seal(
            arg0_cipher->get_elements(), out0_cipher->get_elements(), type,
            m_he_seal_backend, out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        ngraph::he::negate_seal(arg0_plain->get_elements(),
                                out0_plain->get_elements(), type,
                                out0_plain->get_batched_element_count());
      } else {
        throw ngraph_error("Negative types not supported.");
      }
      break;
    }
    case OP_TYPEID::Parameter:
      NGRAPH_INFO << "Skipping parameter";
      break;
    case OP_TYPEID::Pad: {
      const op::Pad* pad = static_cast<const op::Pad*>(&node);
      const Shape arg0_shape = packed_arg_shapes[0];

      if (arg0_cipher != nullptr && arg1_cipher != nullptr &&
          out0_cipher != nullptr) {
        ngraph::he::pad_seal(
            arg0_cipher->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), arg0_shape, packed_out_shape,
            pad->get_padding_below(), pad->get_padding_above(),
            pad->get_pad_mode(), m_batch_size, m_he_seal_backend);
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        ngraph::he::pad_seal(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), arg0_shape, packed_out_shape,
            pad->get_padding_below(), pad->get_padding_above(),
            pad->get_pad_mode(), m_batch_size, m_he_seal_backend);
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        ngraph::he::pad_seal(
            arg0_plain->get_elements(), arg1_plain->get_elements(),
            out0_plain->get_elements(), arg0_shape, packed_out_shape,
            pad->get_padding_below(), pad->get_padding_above(),
            pad->get_pad_mode(), m_batch_size, m_he_seal_backend);
      } else {
        throw ngraph_error("Pad cipher vs. plain types not supported.");
      }
      break;
    }
    case OP_TYPEID::Passthrough: {
      const op::Passthrough* passthrough =
          static_cast<const op::Passthrough*>(&node);
      throw unsupported_op{"Unsupported operation language: " +
                           passthrough->language()};
    }
    case OP_TYPEID::Relu: {
      if (arg0_plain != nullptr && out0_plain != nullptr) {
        size_t output_size = arg0_plain->get_batched_element_count();
        NGRAPH_CHECK(output_size == arg0_plain->num_plaintexts(),
                     "output size ", output_size,
                     " doesn't match number of elements",
                     out0_plain->num_plaintexts());
        ngraph::he::relu_seal(arg0_plain->get_elements(),
                              out0_plain->get_elements(), output_size);
        break;
      }

      if (arg0_cipher == nullptr || out0_cipher == nullptr) {
        throw ngraph_error("Relu types not supported");
      }

      if (!m_enable_client) {
        NGRAPH_WARN
            << "Performing Relu without client is not privacy-preserving";
        size_t output_size = arg0_cipher->get_batched_element_count();
        NGRAPH_CHECK(output_size == arg0_cipher->num_ciphertexts(),
                     "output size ", output_size,
                     " doesn't match number of elements",
                     out0_cipher->num_ciphertexts());
        ngraph::he::relu_seal(arg0_cipher->get_elements(),
                              out0_cipher->get_elements(), output_size,
                              m_he_seal_backend);
        break;
      }

      handle_server_relu_op(arg0_cipher, out0_cipher, node_wrapper);
      break;
    }
    case OP_TYPEID::Reshape: {
      const op::Reshape* reshape = static_cast<const op::Reshape*>(&node);
      Shape op_in_shape;
      Shape op_out_shape;

      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        op_in_shape = arg0_cipher->get_packed_shape();
        op_out_shape = packed_out_shape;
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        op_in_shape = arg0_plain->is_packed() ? arg0_plain->get_packed_shape()
                                              : arg0_plain->get_shape();
        op_out_shape = arg0_plain->is_packed() ? packed_out_shape
                                               : out0_plain->get_shape();
      }

      if (verbose) {
        NGRAPH_INFO << join(op_in_shape, "x") << " reshape "
                    << join(op_out_shape, "x");
      }

      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        ngraph::he::reshape_seal(arg0_cipher->get_elements(),
                                 out0_cipher->get_elements(), op_in_shape,
                                 reshape->get_input_order(), op_out_shape);
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        ngraph::he::reshape_seal(arg0_plain->get_elements(),
                                 out0_plain->get_elements(), op_in_shape,
                                 reshape->get_input_order(), op_out_shape);
      } else {
        throw ngraph_error("Reshape types not supported.");
      }
      break;
    }
    case OP_TYPEID::Result: {
      size_t output_size;
      if (arg0_plain != nullptr) {
        output_size = arg0_plain->get_batched_element_count();
      } else if (arg0_cipher != nullptr) {
        output_size = arg0_cipher->get_batched_element_count();
      } else {
        throw ngraph_error(
            "Input argument is neither plaintext nor ciphertext");
      }
      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        ngraph::he::result_seal(arg0_cipher->get_elements(),
                                out0_cipher->get_elements(), output_size);
      } else if (arg0_plain != nullptr && out0_cipher != nullptr) {
        ngraph::he::result_seal(arg0_plain->get_elements(),
                                out0_cipher->get_elements(), output_size,
                                m_he_seal_backend);
      } else if (arg0_cipher != nullptr && out0_plain != nullptr) {
        ngraph::he::result_seal(arg0_cipher->get_elements(),
                                out0_plain->get_elements(), output_size,
                                m_he_seal_backend);
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        ngraph::he::result_seal(arg0_plain->get_elements(),
                                out0_plain->get_elements(), output_size);
      } else {
        throw ngraph_error("Result types not supported.");
      }
      break;
    }

    case OP_TYPEID::Reverse: {
      const op::Reverse* reverse = static_cast<const op::Reverse*>(&node);
      Shape in_shape = node.get_input_shape(0);

      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        ngraph::he::reverse_seal(arg0_cipher->get_elements(),
                                 out0_cipher->get_elements(), in_shape,
                                 out_shape, reverse->get_reversed_axes());
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        ngraph::he::reverse_seal(arg0_plain->get_elements(),
                                 out0_plain->get_elements(), in_shape,
                                 out_shape, reverse->get_reversed_axes());
      } else {
        throw ngraph_error("Reverse types not supported.");
      }
      break;
    }
    case OP_TYPEID::ScalarConstantLike:
      break;
    case OP_TYPEID::Slice: {
      const op::Slice* slice = static_cast<const op::Slice*>(&node);
      Shape& in_shape = packed_arg_shapes[0];
      Coordinate lower_bounds = slice->get_lower_bounds();
      Coordinate upper_bounds = slice->get_upper_bounds();

      if (m_pack_data) {
        in_shape = unpacked_arg_shapes[0];
        lower_bounds =
            ngraph::he::HETensor::pack_shape(slice->get_lower_bounds());
        upper_bounds =
            ngraph::he::HETensor::pack_shape(slice->get_upper_bounds());
      }

      const Strides& strides = slice->get_strides();

      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        ngraph::he::slice_seal(
            arg0_cipher->get_elements(), out0_cipher->get_elements(), in_shape,
            lower_bounds, upper_bounds, strides, packed_out_shape);
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        for (const auto& elem : arg0_plain->get_elements()) {
          if (elem.num_values() == 0) {
            throw ngraph_error("Slice input has 0 values");
          }
        }
        ngraph::he::slice_seal(
            arg0_plain->get_elements(), out0_plain->get_elements(), in_shape,
            lower_bounds, upper_bounds, strides, packed_out_shape);
      } else {
        throw ngraph_error("Slice types not supported.");
      }
      break;
    }
    case OP_TYPEID::Subtract: {
      if (arg0_cipher != nullptr && arg1_cipher != nullptr &&
          out0_cipher != nullptr) {
        ngraph::he::subtract_seal(
            arg0_cipher->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), type, m_he_seal_backend,
            out0_cipher->get_batched_element_count());
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        ngraph::he::subtract_seal(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), type, m_he_seal_backend,
            out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && arg1_cipher != nullptr &&
                 out0_cipher != nullptr) {
        ngraph::he::subtract_seal(
            arg0_plain->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), type, m_he_seal_backend,
            out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        ngraph::he::subtract_seal(
            arg0_plain->get_elements(), arg1_plain->get_elements(),
            out0_plain->get_elements(), type, m_he_seal_backend,
            out0_plain->get_batched_element_count());
      } else {
        throw ngraph_error("Subtract types not supported.");
      }
      break;
    }
    case OP_TYPEID::Sum: {
      const op::Sum* sum = static_cast<const op::Sum*>(&node);
      Shape op_in_shape = unpacked_arg_shapes[0];

      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        ngraph::he::sum_seal(arg0_cipher->get_elements(),
                             out0_cipher->get_elements(), op_in_shape,
                             out_shape, sum->get_reduction_axes(), type,
                             m_he_seal_backend);
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        ngraph::he::sum_seal(
            arg0_plain->get_elements(), out0_plain->get_elements(), op_in_shape,
            out_shape, sum->get_reduction_axes(), type, m_he_seal_backend);
      } else {
        throw ngraph_error("Sum types not supported.");
      }
      break;
    }
    // Unsupported ops
    case OP_TYPEID::Abs:
    case OP_TYPEID::Acos:
    case OP_TYPEID::All:
    case OP_TYPEID::AllReduce:
    case OP_TYPEID::And:
    case OP_TYPEID::Any:
    case OP_TYPEID::ArgMax:
    case OP_TYPEID::ArgMin:
    case OP_TYPEID::Asin:
    case OP_TYPEID::Atan:
    case OP_TYPEID::AvgPoolBackprop:
    case OP_TYPEID::BatchMatMul:
    case OP_TYPEID::BatchNormTraining:
    case OP_TYPEID::BatchNormTrainingBackprop:
    case OP_TYPEID::BroadcastDistributed:
    case OP_TYPEID::Ceiling:
    case OP_TYPEID::Convert:
    case OP_TYPEID::ConvolutionBackpropData:
    case OP_TYPEID::ConvolutionBackpropFilters:
    case OP_TYPEID::Cos:
    case OP_TYPEID::Cosh:
    case OP_TYPEID::Dequantize:
    case OP_TYPEID::Divide:
    case OP_TYPEID::DynBroadcast:
    case OP_TYPEID::DynPad:
    case OP_TYPEID::DynReshape:
    case OP_TYPEID::DynSlice:
    case OP_TYPEID::DynReplaceSlice:
    case OP_TYPEID::EmbeddingLookup:
    case OP_TYPEID::Equal:
    case OP_TYPEID::Erf:
    case OP_TYPEID::Exp:
    case OP_TYPEID::Floor:
    case OP_TYPEID::Gather:
    case OP_TYPEID::GatherND:
    case OP_TYPEID::GenerateMask:
    case OP_TYPEID::GetOutputElement:
    case OP_TYPEID::Greater:
    case OP_TYPEID::GreaterEq:
    case OP_TYPEID::Less:
    case OP_TYPEID::LessEq:
    case OP_TYPEID::Log:
    case OP_TYPEID::LRN:
    case OP_TYPEID::Max:
    case OP_TYPEID::Maximum:
    case OP_TYPEID::MaxPoolBackprop:
    case OP_TYPEID::Min:
    case OP_TYPEID::Not:
    case OP_TYPEID::NotEqual:
    case OP_TYPEID::OneHot:
    case OP_TYPEID::Or:
    case OP_TYPEID::Power:
    case OP_TYPEID::Product:
    case OP_TYPEID::Quantize:
    case OP_TYPEID::QuantizedAvgPool:
    case OP_TYPEID::QuantizedConvolutionBias:
    case OP_TYPEID::QuantizedConvolutionBiasAdd:
    case OP_TYPEID::QuantizedConvolutionBiasSignedAdd:
    case OP_TYPEID::QuantizedConvolutionRelu:
    case OP_TYPEID::QuantizedConvolution:
    case OP_TYPEID::QuantizedDot:
    case OP_TYPEID::QuantizedDotBias:
    case OP_TYPEID::QuantizedMaxPool:
    case OP_TYPEID::Send:
    case OP_TYPEID::Recv:
    case OP_TYPEID::Range:
    case OP_TYPEID::ReluBackprop:
    case OP_TYPEID::ReplaceSlice:
    case OP_TYPEID::ReverseSequence:
    case OP_TYPEID::ScatterAdd:
    case OP_TYPEID::ScatterNDAdd:
    case OP_TYPEID::Select:
    case OP_TYPEID::ShapeOf:
    case OP_TYPEID::Sigmoid:
    case OP_TYPEID::SigmoidBackprop:
    case OP_TYPEID::Sign:
    case OP_TYPEID::Sin:
    case OP_TYPEID::Sinh:
    case OP_TYPEID::Softmax:
    case OP_TYPEID::Sqrt:
    case OP_TYPEID::StopGradient:
    case OP_TYPEID::Tan:
    case OP_TYPEID::Tanh:
    case OP_TYPEID::Tile:
    case OP_TYPEID::TopK:
    case OP_TYPEID::Transpose:
    case OP_TYPEID::Xor:
    default:
      throw unsupported_op("Unsupported op '" + node.description() + "'");
#pragma GCC diagnostic pop
  }
}

void ngraph::he::HESealExecutable::handle_server_max_pool_op(
    std::shared_ptr<HESealCipherTensor>& arg0_cipher,
    std::shared_ptr<HESealCipherTensor>& out_cipher,
    const NodeWrapper& node_wrapper) {
  const Node& node = *node_wrapper.get_node();
  bool verbose = verbose_op(node);
  const op::MaxPool* max_pool = static_cast<const op::MaxPool*>(&node);

  NGRAPH_INFO << "Handling server maxpool op";

  m_max_pool_done = false;

  Shape unpacked_arg_shape = node.get_input_shape(0);
  Shape packed_out_shape =
      ngraph::he::HETensor::pack_shape(node.get_output_shape(0));

  NGRAPH_INFO << "unpacked_arg_shape " << join(unpacked_arg_shape, "x");
  NGRAPH_INFO << "packed_out_shape " << join(packed_out_shape, "x");

  std::vector<std::vector<size_t>> maximize_list = ngraph::he::max_pool_seal(
      unpacked_arg_shape, packed_out_shape, max_pool->get_window_shape(),
      max_pool->get_window_movement_strides(), max_pool->get_padding_below(),
      max_pool->get_padding_above());

  NGRAPH_INFO << "maximize_list.size " << maximize_list.size();

  m_max_pool_ciphertexts.clear();

  for (size_t list_ind = 0; list_ind < maximize_list.size(); list_ind++) {
    he_proto::TCPMessage proto_msg;
    proto_msg.set_type(he_proto::TCPMessage_Type_REQUEST);

    NGRAPH_INFO << "List in " << list_ind;

    json js;
    js["function"] = node.description();
    NGRAPH_INFO << "Description " << js["function"];

    he_proto::Function f;
    f.set_function(js.dump());
    *proto_msg.mutable_function() = f;

    for (const size_t max_ind : maximize_list[list_ind]) {
      arg0_cipher->get_element(max_ind)->save(*proto_msg.add_ciphers());
    }

    // Send list of ciphertexts to maximize over to client
    if (verbose) {
      NGRAPH_INFO << "Sending " << proto_msg.ciphers_size()
                  << " Maxpool ciphertexts to client";
    }

    ngraph::he::TCPMessage max_pool_message(proto_msg);
    m_session->write_message(std::move(max_pool_message));

    // Acquire lock
    std::unique_lock<std::mutex> mlock(m_max_pool_mutex);

    // Wait until max is done
    m_max_pool_cond.wait(mlock,
                         std::bind(&HESealExecutable::max_pool_done, this));

    // Reset for next max_pool call
    m_max_pool_done = false;
  }
  NGRAPH_INFO << "Done with maxpool calling; setting elements";
  out_cipher->set_elements(m_max_pool_ciphertexts);
  NGRAPH_INFO << "Done setting maxpool elements";
}

void ngraph::he::HESealExecutable::handle_server_relu_op(
    std::shared_ptr<HESealCipherTensor>& arg_cipher,
    std::shared_ptr<HESealCipherTensor>& out_cipher,
    const NodeWrapper& node_wrapper) {
  auto type_id = node_wrapper.get_typeid();
  NGRAPH_CHECK(type_id == OP_TYPEID::Relu || type_id == OP_TYPEID::BoundedRelu,
               "only support relu / bounded relu");

  const Node& node = *node_wrapper.get_node();
  bool verbose = verbose_op(node);
  size_t element_count = shape_size(node.get_output_shape(0)) / m_batch_size;

  if (arg_cipher == nullptr || out_cipher == nullptr) {
    NGRAPH_INFO << "Relu types not supported ";
    throw ngraph_error("Relu types not supported.");
  }

  size_t smallest_ind = ngraph::he::match_to_smallest_chain_index(
      arg_cipher->get_elements(), m_he_seal_backend);

  if (verbose) {
    NGRAPH_INFO << "Matched moduli to chain ind " << smallest_ind;
  }

  m_relu_ciphertexts.clear();
  m_relu_ciphertexts.resize(element_count);
  for (size_t relu_idx = 0; relu_idx < element_count; ++relu_idx) {
    m_relu_ciphertexts[relu_idx] = std::make_shared<SealCiphertextWrapper>();
  }

  // TODO: tune
  const size_t max_relu_message_cnt = 100;

  m_unknown_relu_idx.clear();
  m_unknown_relu_idx.reserve(element_count);

  // TODO: don't just use 6
  float alpha = 6.0f;

  // Process known values
  for (size_t relu_idx = 0; relu_idx < element_count; ++relu_idx) {
    auto& cipher = *arg_cipher->get_element(relu_idx);
    if (cipher.known_value()) {
      if (type_id == OP_TYPEID::Relu) {
        ngraph::he::scalar_relu_seal_known_value(cipher,
                                                 m_relu_ciphertexts[relu_idx]);
      } else {
        ngraph::he::scalar_bounded_relu_seal_known_value(
            cipher, m_relu_ciphertexts[relu_idx], alpha);
      }
    } else {
      m_unknown_relu_idx.emplace_back(relu_idx);
    }
  }
  auto process_unknown_relu_ciphers_batch =
      [&](const std::vector<std::shared_ptr<SealCiphertextWrapper>>&
              cipher_batch) {
        if (verbose) {
          NGRAPH_INFO << "Sending relu request size " << cipher_batch.size();
        }

        he_proto::TCPMessage proto_msg;
        proto_msg.set_type(he_proto::TCPMessage_Type_REQUEST);

        json js;
        js["function"] = node.description();
        NGRAPH_INFO << "Description " << js["function"];

        he_proto::Function f;
        f.set_function(js.dump());
        *proto_msg.mutable_function() = f;

        for (size_t cipher_idx = 0; cipher_idx < cipher_batch.size();
             ++cipher_idx) {
          proto_msg.add_ciphers();
        }
#pragma omp parallel for
        for (size_t cipher_idx = 0; cipher_idx < cipher_batch.size();
             ++cipher_idx) {
          cipher_batch[cipher_idx]->save(
              *proto_msg.mutable_ciphers(cipher_idx));
        }

        ngraph::he::TCPMessage relu_message(proto_msg);
        m_session->write_message(std::move(relu_message));
      };

  // Process unknown values
  std::vector<std::shared_ptr<SealCiphertextWrapper>> relu_ciphers_batch;
  relu_ciphers_batch.reserve(max_relu_message_cnt);

  for (const auto& unknown_relu_idx : m_unknown_relu_idx) {
    auto& cipher = arg_cipher->get_element(unknown_relu_idx);
    relu_ciphers_batch.emplace_back(cipher);
    if (relu_ciphers_batch.size() == max_relu_message_cnt) {
      process_unknown_relu_ciphers_batch(relu_ciphers_batch);
      relu_ciphers_batch.clear();
    }
  }
  if (relu_ciphers_batch.size() != 0) {
    process_unknown_relu_ciphers_batch(relu_ciphers_batch);
    relu_ciphers_batch.clear();
  }

  // Wait until all batches have been processed
  std::unique_lock<std::mutex> mlock(m_relu_mutex);
  m_relu_cond.wait(
      mlock, [=]() { return m_relu_done_count == m_unknown_relu_idx.size(); });
  m_relu_done_count = 0;

  out_cipher->set_elements(m_relu_ciphertexts);
}
