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

#include "seal/he_seal_executable.hpp"

#include <functional>
#include <limits>
#include <tuple>
#include <unordered_set>

#include "he_op_annotations.hpp"
#include "he_tensor.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/passthrough.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
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
#include "pass/propagate_he_annotations.hpp"
#include "pass/supported_ops.hpp"
#include "protos/message.pb.h"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/add_seal.hpp"
#include "seal/kernel/avg_pool_seal.hpp"
#include "seal/kernel/batch_norm_inference_seal.hpp"
#include "seal/kernel/bounded_relu_seal.hpp"
#include "seal/kernel/broadcast_seal.hpp"
#include "seal/kernel/concat_seal.hpp"
#include "seal/kernel/constant_seal.hpp"
#include "seal/kernel/convolution_seal.hpp"
#include "seal/kernel/divide_seal.hpp"
#include "seal/kernel/dot_seal.hpp"
#include "seal/kernel/exp_seal.hpp"
#include "seal/kernel/max_pool_seal.hpp"
#include "seal/kernel/max_seal.hpp"
#include "seal/kernel/minimum_seal.hpp"
#include "seal/kernel/multiply_seal.hpp"
#include "seal/kernel/negate_seal.hpp"
#include "seal/kernel/pad_seal.hpp"
#include "seal/kernel/power_seal.hpp"
#include "seal/kernel/relu_seal.hpp"
#include "seal/kernel/rescale_seal.hpp"
#include "seal/kernel/reshape_seal.hpp"
#include "seal/kernel/result_seal.hpp"
#include "seal/kernel/reverse_seal.hpp"
#include "seal/kernel/slice_seal.hpp"
#include "seal/kernel/softmax_seal.hpp"
#include "seal/kernel/subtract_seal.hpp"
#include "seal/kernel/sum_seal.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"

using json = nlohmann::json;
using ngraph::descriptor::layout::DenseTensorLayout;

namespace ngraph {
namespace he {
HESealExecutable::HESealExecutable(const std::shared_ptr<Function>& function,
                                   bool enable_performance_collection,
                                   HESealBackend& he_seal_backend)
    : m_he_seal_backend(he_seal_backend),
      m_enable_client{enable_client},
      m_batch_size{1},
      m_port{34000} {
  // TODO(fboemer): Use
  (void)enable_performance_collection;  // Avoid unused parameter warning

  m_context = he_seal_backend.get_context();
  // TODO(fboemer): use clone_function? (check
  // https://github.com/NervanaSystems/ngraph/pull/3773 is merged)
  m_function = function;

  NGRAPH_HE_LOG(3) << "Creating Executable";
  for (const auto& param : m_function->get_parameters()) {
    NGRAPH_HE_LOG(3) << "Parameter " << param->get_name();
    if (HEOpAnnotations::has_he_annotation(*param)) {
      std::string from_client_str = from_client(*param) ? "" : "not ";
      NGRAPH_HE_LOG(3) << "\tshape " << param->get_shape() << " is "
                       << from_client_str << "from client";
    }

    for (const auto& tag : param->get_provenance_tags()) {
      NGRAPH_HE_LOG(3) << "\tTag " << tag;
    }
  }

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

  NGRAPH_HE_LOG(3) << "Running optimization passes";
  ngraph::pass::Manager pass_manager;
  pass_manager.set_pass_visualization(false);
  pass_manager.set_pass_serialization(false);

  pass_manager.register_pass<ngraph::pass::LikeReplacement>();
  pass_manager.register_pass<ngraph::pass::AssignLayout<DenseTensorLayout>>();
  pass_manager.register_pass<ngraph::pass::CoreFusion>();
  if (!m_stop_const_fold) {
    NGRAPH_HE_LOG(4) << "Registering constant folding pass";
    pass_manager.register_pass<ngraph::pass::ConstantFolding>();
  }

  NGRAPH_HE_LOG(4) << "Running passes";
  pass_manager.run_passes(m_function);

  ngraph::pass::Manager pass_manager_he;
  pass_manager_he.set_pass_visualization(false);
  pass_manager_he.set_pass_serialization(false);
  pass_manager_he.register_pass<pass::HEFusion>();
  pass_manager_he.register_pass<pass::HELiveness>();
  pass_manager_he.register_pass<pass::SupportedOps>(
      [this](const ngraph::Node& op) {
        return m_he_seal_backend.is_supported(op);
      });

  NGRAPH_HE_LOG(4) << "Running HE passes";
  pass_manager_he.run_passes(m_function);

  update_he_op_annotations();
}

HESealExecutable::~HESealExecutable() {
  NGRAPH_HE_LOG(3) << "~HESealExecutable()";
  if (m_server_setup) {
    NGRAPH_HE_LOG(5) << "Waiting for m_message_handling_thread to join";
    m_message_handling_thread.join();
    NGRAPH_HE_LOG(5) << "m_message_handling_thread joined";

    // m_acceptor and m_io_context both free the socket? Avoid double-free
    m_acceptor->close();
    m_acceptor = nullptr;
    m_session = nullptr;
  }
}

void HESealExecutable::update_he_op_annotations() {
  NGRAPH_HE_LOG(3) << "Upadting HE op annotations";
  ngraph::pass::Manager pass_manager_he;
  pass_manager_he.register_pass<pass::PropagateHEAnnotations>();
  pass_manager_he.run_passes(m_function);
  m_is_compiled = true;

  m_wrapped_nodes.clear();
  for (const std::shared_ptr<Node>& node : m_function->get_ordered_ops()) {
    m_wrapped_nodes.emplace_back(node);
  }
  set_parameters_and_results(*m_function);
}

void HESealExecutable::set_batch_size(size_t batch_size) {
  size_t max_batch_size = m_he_seal_backend.get_ckks_encoder()->slot_count();
  if (complex_packing()) {
    max_batch_size *= 2;
  }
  NGRAPH_CHECK(batch_size <= max_batch_size, "Batch size ", batch_size,
               " too large (maximum ", max_batch_size, ")");
  m_batch_size = batch_size;

  NGRAPH_HE_LOG(5) << "Server set batch size to " << m_batch_size;
}

void HESealExecutable::check_client_supports_function() {
  // Check if single parameter is from client
  size_t from_client_count = 0;
  for (const auto& param : get_parameters()) {
    if (from_client(*param)) {
      from_client_count++;
      NGRAPH_HE_LOG(5) << "Parameter " << param->get_name() << " from client";
    }
  }
  NGRAPH_CHECK(get_results().size() == 1,
               "HESealExecutable only supports output size 1 (got ",
               get_results().size(), ")");
  NGRAPH_CHECK(from_client_count > 0, "Expected > 0 parameters from client");
}

bool HESealExecutable::server_setup() {
  if (!m_server_setup) {
    NGRAPH_HE_LOG(1) << "Enable client";

    check_client_supports_function();

    NGRAPH_HE_LOG(1) << "Starting server";
    start_server();

    if (enable_garbled_circuits()) {
      m_aby_executor = std::make_unique<aby::ABYServerExecutor>(
          *this, std::string("yao"), std::string("0.0.0.0"));
    }

    std::stringstream param_stream;
    m_he_seal_backend.get_encryption_parameters().save(param_stream);

    proto::EncryptionParameters proto_parms;
    *proto_parms.mutable_encryption_parameters() = param_stream.str();

    proto::TCPMessage proto_msg;
    *proto_msg.mutable_encryption_parameters() = proto_parms;
    proto_msg.set_type(proto::TCPMessage_Type_RESPONSE);

    TCPMessage parms_message(std::move(proto_msg));
    NGRAPH_HE_LOG(3) << "Server waiting until session started";
    std::unique_lock<std::mutex> mlock(m_session_mutex);
    m_session_cond.wait(mlock, [this]() { return this->session_started(); });

    NGRAPH_HE_LOG(3) << "Server writing parameters message";
    m_session->write_message(std::move(parms_message));
    m_server_setup = true;

    // Set client inputs to dummy values
    if (m_is_compiled) {
      m_client_inputs.clear();
      m_client_inputs.resize(get_parameters().size());
    } else {
      NGRAPH_HE_LOG(1) << "Client already setup";
    }
  }
  return true;
}

void HESealExecutable::accept_connection() {
  NGRAPH_HE_LOG(1) << "Server accepting connections";
  auto server_callback =
      std::bind(&HESealExecutable::handle_message, this, std::placeholders::_1);

  m_acceptor->async_accept([this, server_callback](boost::system::error_code ec,
                                                   tcp::socket socket) {
    if (!ec) {
      NGRAPH_HE_LOG(1) << "Connection accepted";
      m_session =
          std::make_shared<TCPSession>(std::move(socket), server_callback);
      m_session->start();
      NGRAPH_HE_LOG(1) << "Session started";

      std::lock_guard<std::mutex> guard(m_session_mutex);
      m_session_started = true;
      m_session_cond.notify_one();
    } else {
      NGRAPH_ERR << "error accepting connection " << ec.message();
      accept_connection();
    }
  });
}

void HESealExecutable::start_server() {
  tcp::resolver resolver(m_io_context);
  tcp::endpoint server_endpoints(tcp::v4(), m_port);
  m_acceptor = std::make_unique<tcp::acceptor>(m_io_context, server_endpoints);
  boost::asio::socket_base::reuse_address option(true);
  m_acceptor->set_option(option);

  accept_connection();
  m_message_handling_thread = std::thread([this]() {
    try {
      m_io_context.run();
    } catch (std::exception& e) {
      NGRAPH_ERR << "Server error hanndling thread: " << std::string(e.what());
      NGRAPH_CHECK(false, "Server error hanndling thread: ", e.what());
    };
  });
}

void HESealExecutable::load_public_key(const proto::TCPMessage& proto_msg) {
  NGRAPH_CHECK(proto_msg.has_public_key(), "proto_msg doesn't have public key");

  seal::PublicKey key;
  const std::string& pk_str = proto_msg.public_key().public_key();
  std::stringstream key_stream(pk_str);
  key.load(m_context, key_stream);
  m_he_seal_backend.set_public_key(key);

  m_client_public_key_set = true;
}

void HESealExecutable::load_eval_key(const proto::TCPMessage& proto_msg) {
  NGRAPH_CHECK(proto_msg.has_eval_key(), "proto_msg doesn't have eval key");

  seal::RelinKeys keys;
  const std::string& evk_str = proto_msg.eval_key().eval_key();
  std::stringstream key_stream(evk_str);
  keys.load(m_context, key_stream);
  m_he_seal_backend.set_relin_keys(keys);

  m_client_eval_key_set = true;
}

void HESealExecutable::send_inference_shape() {
  m_sent_inference_shape = true;

  const ParameterVector& input_parameters = get_parameters();

  proto::TCPMessage proto_msg;
  proto_msg.set_type(proto::TCPMessage_Type_REQUEST);

  for (const auto& input_param : input_parameters) {
    if (from_client(*input_param)) {
      proto::HETensor* proto_he_tensor = proto_msg.add_he_tensors();

      std::vector<uint64_t> shape{input_param->get_shape()};
      *proto_he_tensor->mutable_shape() = {shape.begin(), shape.end()};

      std::string name = input_param->get_provenance_tags().empty()
                             ? input_param->get_name()
                             : *input_param->get_provenance_tags().begin();

      NGRAPH_HE_LOG(1) << "Server setting inference tensor name " << name
                       << " (corresponding to Parameter "
                       << input_param->get_name() << "), with "
                       << input_param->get_shape();

      proto_he_tensor->set_name(name);

      if (plaintext_packed(*input_param)) {
        NGRAPH_HE_LOG(1) << "Setting parameter " << input_param->get_name()
                         << " to packed";
        proto_he_tensor->set_packed(true);
      }
    }
  }

  NGRAPH_HE_LOG(1) << "Server sending inference of "
                   << proto_msg.he_tensors_size() << " parameters";

  json js = {{"function", "Parameter"}};
  proto::Function f;
  f.set_function(js.dump());
  NGRAPH_HE_LOG(3) << "js " << js.dump();
  *proto_msg.mutable_function() = f;

  TCPMessage execute_msg(std::move(proto_msg));
  m_session->write_message(std::move(execute_msg));
}

void HESealExecutable::handle_relu_result(const proto::TCPMessage& proto_msg) {
  NGRAPH_HE_LOG(3) << "Server handling relu result";
  std::lock_guard<std::mutex> guard(m_relu_mutex);

  NGRAPH_CHECK(proto_msg.he_tensors_size() == 1,
               "Can only handle one tensor at a time, got ",
               proto_msg.he_tensors_size());

  const auto& proto_tensor = proto_msg.he_tensors(0);
  auto he_tensor = HETensor::load_from_proto_tensor(
      proto_tensor, *m_he_seal_backend.get_ckks_encoder(),
      m_he_seal_backend.get_context(), *m_he_seal_backend.get_encryptor(),
      *m_he_seal_backend.get_decryptor(),
      m_he_seal_backend.get_encryption_parameters());

  size_t result_count = proto_tensor.data_size();
  for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
    m_relu_data[m_unknown_relu_idx[result_idx + m_relu_done_count]] =
        he_tensor->data(result_idx);
  }

  if (enable_garbled_circuits()) {
    NGRAPH_INFO << "Performing garbled circuits output mask correction";
    m_aby_executor->post_process_aby_circuit(proto_msg.function().function(),
                                             he_tensor);
  }

  m_relu_done_count += result_count;
  m_relu_cond.notify_all();
}

void HESealExecutable::handle_bounded_relu_result(
    const proto::TCPMessage& proto_msg) {
  handle_relu_result(proto_msg);
}

void HESealExecutable::handle_max_pool_result(
    const proto::TCPMessage& proto_msg) {
  std::lock_guard<std::mutex> guard(m_max_pool_mutex);

  NGRAPH_CHECK(proto_msg.he_tensors_size() == 1,
               "Can only handle one tensor at a time, got ",
               proto_msg.he_tensors_size());

  const auto& proto_tensor = proto_msg.he_tensors(0);
  size_t result_count = proto_tensor.data_size();

  NGRAPH_CHECK(result_count == 1, "Maxpool only supports result_count 1, got ",
               result_count);

  auto he_tensor = HETensor::load_from_proto_tensor(
      proto_tensor, *m_he_seal_backend.get_ckks_encoder(),
      m_he_seal_backend.get_context(), *m_he_seal_backend.get_encryptor(),
      *m_he_seal_backend.get_decryptor(),
      m_he_seal_backend.get_encryption_parameters());

  m_max_pool_data.emplace_back(he_tensor->data(0));
  m_max_pool_done = true;
  m_max_pool_cond.notify_all();
}

void HESealExecutable::handle_message(const TCPMessage& message) {
  NGRAPH_HE_LOG(3) << "Server handling message";
  std::shared_ptr<proto::TCPMessage> proto_msg = message.proto_message();

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-enum"
  switch (proto_msg->type()) {
    case proto::TCPMessage_Type_RESPONSE: {
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
          throw ngraph_error("Unknown function name");
        }
      }
      break;
    }
    case proto::TCPMessage_Type_REQUEST: {
      if (proto_msg->he_tensors_size() > 0) {
        handle_client_ciphers(*proto_msg);
      }
      break;
    }
    case proto::TCPMessage_Type_UNKNOWN:
    default:
      NGRAPH_CHECK(false, "Unknonwn TCPMessage type");
  }
#pragma clang diagnostic pop
}

void HESealExecutable::handle_client_ciphers(
    const proto::TCPMessage& proto_msg) {
  NGRAPH_HE_LOG(3) << "Handling client tensors";

  NGRAPH_CHECK(proto_msg.he_tensors_size() > 0,
               "Client received empty tensor message");
  NGRAPH_CHECK(proto_msg.he_tensors_size() == 1,
               "Client only supports 1 client tensor");
  // TODO(fboemer): check for uniqueness of batch size if > 1 input tensor

  const ParameterVector& input_parameters = get_parameters();

  /// \brief Looks for a parameter which matches a given tensor name
  /// \param[in] tensor_name Tensor name to match against
  /// \param[out] matching_idx Will be populated if a match is found
  /// \returns Whether or not a matching parameter shape has been found
  auto find_matching_parameter_index = [&](const std::string& tensor_name,
                                           size_t& matching_idx) {
    NGRAPH_HE_LOG(5) << "Calling find_matching_parameter_index(" << tensor_name
                     << ")";
    for (size_t param_idx = 0; param_idx < input_parameters.size();
         ++param_idx) {
      const auto& parameter = input_parameters[param_idx];

      for (const auto& tag : parameter->get_provenance_tags()) {
        NGRAPH_HE_LOG(5) << "Tag " << tag;
      }

      if (param_originates_from_name(*parameter, tensor_name)) {
        NGRAPH_HE_LOG(5) << "Param " << tensor_name << " matches at index "
                         << param_idx;
        matching_idx = param_idx;
        return true;
      }
    }
    NGRAPH_HE_LOG(5) << "Could not find tensor " << tensor_name;
    return false;
  };

  auto& proto_tensor = proto_msg.he_tensors(0);
  ngraph::Shape shape{proto_tensor.shape().begin(), proto_tensor.shape().end()};

  NGRAPH_HE_LOG(5) << "proto_tensor.packed() " << proto_tensor.packed();
  set_batch_size(HETensor::batch_size(shape, proto_tensor.packed()));
  NGRAPH_HE_LOG(5) << "Offset " << proto_tensor.offset();

  size_t param_idx;
  NGRAPH_CHECK(find_matching_parameter_index(proto_tensor.name(), param_idx),
               "Could not find matching parameter name ", proto_tensor.name());

  if (m_client_inputs[param_idx] == nullptr) {
    auto he_tensor = HETensor::load_from_proto_tensor(
        proto_tensor, *m_he_seal_backend.get_ckks_encoder(),
        m_he_seal_backend.get_context(), *m_he_seal_backend.get_encryptor(),
        *m_he_seal_backend.get_decryptor(),
        m_he_seal_backend.get_encryption_parameters());
    m_client_inputs[param_idx] = he_tensor;
  } else {
    HETensor::load_from_proto_tensor(m_client_inputs[param_idx], proto_tensor,
                                     m_he_seal_backend.get_context());
  }

  auto done_loading = [&]() {
    for (size_t parm_idx = 0; parm_idx < input_parameters.size(); ++parm_idx) {
      const auto& param = input_parameters[parm_idx];
      if (from_client(*param)) {
        NGRAPH_HE_LOG(5) << "From client param shape " << param->get_shape();
        NGRAPH_HE_LOG(5) << "m_batch_size " << m_batch_size;

        if (m_client_inputs[parm_idx] == nullptr ||
            !m_client_inputs[parm_idx]->done_loading()) {
          return false;
        }
      }
    }
    return true;
  };

  if (done_loading()) {
    NGRAPH_HE_LOG(3) << "Done loading client ciphertexts";

    std::lock_guard<std::mutex> guard(m_client_inputs_mutex);
    m_client_inputs_received = true;
    NGRAPH_HE_LOG(5) << "Notifying done loading client ciphertexts";
    m_client_inputs_cond.notify_all();
  } else {
    NGRAPH_HE_LOG(3) << "Not yet done loading client ciphertexts";
  }
}

std::vector<ngraph::runtime::PerformanceCounter>
HESealExecutable::get_performance_data() const {
  std::vector<runtime::PerformanceCounter> rc;
  for (const auto& [node, stop_watch] : m_timer_map) {
    rc.emplace_back(node, stop_watch.get_total_microseconds(),
                    stop_watch.get_call_count());
  }
  return rc;
}

bool HESealExecutable::call(
    const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& server_inputs) {
  NGRAPH_HE_LOG(3) << "HESealExecutable::call";
  validate(outputs, server_inputs);
  NGRAPH_HE_LOG(3) << "HESealExecutable::call validated inputs";

  if (enable_client()) {
    if (!server_setup()) {
      return false;
    }
  }

  if (complex_packing()) {
    NGRAPH_HE_LOG(1) << "Complex packing";
  }

  if (enable_client()) {
    NGRAPH_HE_LOG(1) << "Waiting for m_client_inputs";

    std::unique_lock<std::mutex> mlock(m_client_inputs_mutex);
    m_client_inputs_cond.wait(
        mlock, std::bind(&HESealExecutable::client_inputs_received, this));
    NGRAPH_HE_LOG(1) << "Client inputs_received";
  }

  // convert inputs to HETensor
  NGRAPH_HE_LOG(3) << "Converting inputs to HETensor";
  const auto& parameters = get_parameters();
  std::vector<std::shared_ptr<HETensor>> he_inputs;
  for (size_t input_idx = 0; input_idx < server_inputs.size(); ++input_idx) {
    auto param_shape = server_inputs[input_idx]->get_shape();
    auto& param = parameters[input_idx];
    std::shared_ptr<HETensor> he_input;

    if (enable_client() && from_client(*param)) {
      NGRAPH_HE_LOG(1) << "Processing parameter " << param->get_name()
                       << "(shape {" << param_shape << "}) from client";
      NGRAPH_CHECK(m_client_inputs.size() > input_idx,
                   "Not enough client inputs");
      he_input = std::static_pointer_cast<HETensor>(m_client_inputs[input_idx]);

      if (auto current_annotation = std::dynamic_pointer_cast<HEOpAnnotations>(
              param->get_op_annotations())) {
        NGRAPH_CHECK(
            current_annotation->packed() == he_input->is_packed(),
            "Parameter annotation ", *current_annotation, " does not match ",
            (he_input->is_packed() ? "packed" : "unpacked"), "input tensor");

        current_annotation->set_encrypted(he_input->any_encrypted_data());
        param->set_op_annotations(current_annotation);

      } else {
        NGRAPH_WARN << "Parameter " << param->get_name()
                    << " has no HE op annotation";
      }
    } else {
      NGRAPH_HE_LOG(1) << "Processing parameter " << param->get_name()
                       << "(shape {" << param_shape << "}) from server";

      auto he_server_input =
          std::static_pointer_cast<HETensor>(server_inputs[input_idx]);
      he_input = he_server_input;

      if (auto current_annotation = std::dynamic_pointer_cast<HEOpAnnotations>(
              param->get_op_annotations())) {
        NGRAPH_HE_LOG(5) << "Parameter " << param->get_name()
                         << " has annotation " << *current_annotation;
        if (!he_input->any_encrypted_data()) {
          if (current_annotation->packed()) {
            he_input->pack();
          } else {
            he_input->unpack();
          }
        }

        if (current_annotation->encrypted()) {
          NGRAPH_HE_LOG(3) << "Encrypting parameter " << param->get_name()
                           << " from server";

#pragma omp parallel for
          for (size_t he_type_idx = 0;
               he_type_idx < he_input->get_batched_element_count();
               ++he_type_idx) {
            if (he_input->data(he_type_idx).is_plaintext()) {
              auto cipher = HESealBackend::create_empty_ciphertext();
              m_he_seal_backend.encrypt(
                  cipher, he_input->data(he_type_idx).get_plaintext(),
                  he_input->get_element_type(),
                  he_input->data(he_type_idx).complex_packing());
              he_input->data(he_type_idx).set_ciphertext(cipher);
            }
          }

          NGRAPH_CHECK(he_input->is_packed() == current_annotation->packed(),
                       "Mismatch between tensor input and annotation (",
                       he_input->is_packed(),
                       " != ", current_annotation->packed(), ")");

          NGRAPH_HE_LOG(3) << "Done encrypting parameter " << param->get_name()
                           << " from server";
        }
      }
    }
    NGRAPH_CHECK(he_input != nullptr, "HE input is nullptr");
    if (he_input->is_packed()) {
      set_batch_size(he_input->get_batch_size());
    }
    he_inputs.emplace_back(he_input);
  }

  NGRAPH_HE_LOG(3) << "Updating HE op annotations";
  update_he_op_annotations();

  NGRAPH_HE_LOG(3) << "Converting outputs to HETensor";
  std::vector<std::shared_ptr<HETensor>> he_outputs;
  he_outputs.reserve(outputs.size());
  for (auto& tensor : outputs) {
    he_outputs.push_back(std::static_pointer_cast<HETensor>(tensor));
  }

  NGRAPH_HE_LOG(3) << "Mapping function parameters to HETensor";
  NGRAPH_CHECK(he_inputs.size() >= parameters.size(),
               "Not enough inputs in input map");
  std::unordered_map<ngraph::descriptor::Tensor*, std::shared_ptr<HETensor>>
      tensor_map;
  size_t input_count = 0;
  for (const auto& param : parameters) {
    for (size_t param_out_idx = 0; param_out_idx < param->get_output_size();
         ++param_out_idx) {
      descriptor::Tensor* tensor =
          param->get_output_tensor_ptr(param_out_idx).get();
      tensor_map.insert({tensor, he_inputs[input_count++]});
    }
  }

  NGRAPH_HE_LOG(3) << "Mapping function outputs to HETensor";
  for (size_t output_count = 0; output_count < get_results().size();
       ++output_count) {
    auto output = get_results()[output_count];
    if (!std::dynamic_pointer_cast<op::Result>(output)) {
      throw ngraph_error("One of function's outputs isn't op::Result");
    }
    ngraph::descriptor::Tensor* tv = output->get_output_tensor_ptr(0).get();

    auto& he_output = he_outputs[output_count];

    if (HEOpAnnotations::has_he_annotation(*output)) {
      auto he_op_annotation = HEOpAnnotations::he_op_annotation(*output);

      if (!he_output->any_encrypted_data()) {
        if (he_op_annotation->packed()) {
          he_output->pack();
        } else {
          he_output->unpack();
        }
      }
    }
    tensor_map.insert({tv, he_output});
  }

  // for each ordered op in the graph
  for (const NodeWrapper& wrapped : m_wrapped_nodes) {
    auto op = wrapped.get_node();
    auto type_id = wrapped.get_typeid();
    bool verbose = verbose_op(*op);

    if (verbose) {
      NGRAPH_HE_LOG(3) << "\033[1;32m"
                       << "[ " << op->get_name() << " ]"
                       << "\033[0m";
      if (type_id == OP_TYPEID::Constant) {
        NGRAPH_HE_LOG(3) << "Constant shape " << op->get_shape();
      }
    }

    if (type_id == OP_TYPEID::Parameter) {
      if (verbose) {
        const auto param_op =
            std::static_pointer_cast<const ngraph::op::Parameter>(op);
        if (HEOpAnnotations::has_he_annotation(*param_op)) {
          std::string from_client_str = from_client(*param_op) ? "" : " not";
          NGRAPH_HE_LOG(3) << "Parameter shape " << param_op->get_shape()
                           << from_client_str << " from client";
        }
      }
      continue;
    }
    m_timer_map[op].start();

    // get op inputs from map
    std::vector<std::shared_ptr<HETensor>> op_inputs;
    for (auto input : op->inputs()) {
      descriptor::Tensor* tensor = &input.get_tensor();
      op_inputs.push_back(tensor_map.at(tensor));
    }

    if (enable_client() && type_id == OP_TYPEID::Result) {
      // Client outputs don't have decryption performed, so skip result op
      NGRAPH_HE_LOG(3) << "Setting client outputs";
      m_client_outputs = op_inputs;
    }

    // get op outputs from map or create
    std::vector<std::shared_ptr<HETensor>> op_outputs;
    for (size_t i = 0; i < op->get_output_size(); ++i) {
      auto tensor = &op->output(i).get_tensor();
      auto it = tensor_map.find(tensor);
      if (it == tensor_map.end()) {
        // The output tensor is not in the tensor map so create a new tensor
        Shape shape = op->get_output_shape(i);
        const element::Type& element_type = op->get_output_element_type(i);
        std::string name = op->output(i).get_tensor().get_name();

        NGRAPH_HE_LOG(3) << "Get output packing / encrypted";

        // TODO(fboemer): remove case once Constant becomes an op
        // (https://github.com/NervanaSystems/ngraph/pull/3752)
        bool encrypted_out;
        bool packed_out;
        if (op->is_op()) {
          std::shared_ptr<HEOpAnnotations> he_op_annotation =
              HEOpAnnotations::he_op_annotation(
                  *std::static_pointer_cast<const ngraph::op::Op>(op));
          encrypted_out = he_op_annotation->encrypted();
          packed_out = he_op_annotation->packed();
        } else {
          NGRAPH_WARN
              << "Node " << op->get_name()
              << " is not op, using default encrypted / packing behavior";
          encrypted_out =
              std::any_of(op_inputs.begin(), op_inputs.end(),
                          [](std::shared_ptr<ngraph::he::HETensor> op_input) {
                            return op_input->any_encrypted_data();
                          });
          packed_out =
              std::any_of(op_inputs.begin(), op_inputs.end(),
                          [](std::shared_ptr<ngraph::he::HETensor> he_tensor) {
                            return he_tensor->is_packed();
                          });
        }
        NGRAPH_HE_LOG(3) << "encrypted_out " << encrypted_out;
        NGRAPH_HE_LOG(3) << "packed_out " << packed_out;
        if (packed_out) {
          HETensor::unpack_shape(shape, m_batch_size);
        }
        NGRAPH_HE_LOG(5) << "Creating output tensor with shape " << shape;

        if (encrypted_out) {
          auto out_tensor = std::static_pointer_cast<HETensor>(
              m_he_seal_backend.create_cipher_tensor(element_type, shape,
                                                     packed_out, name));
          tensor_map.insert({tensor, out_tensor});
        } else {
          auto out_tensor = std::static_pointer_cast<HETensor>(
              m_he_seal_backend.create_plain_tensor(element_type, shape,
                                                    packed_out, name));
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
      NGRAPH_HE_LOG(3) << "\033[1;31m" << op->get_name() << " took "
                       << m_timer_map[op].get_milliseconds() << "ms"
                       << "\033[0m";
    }
  }
  size_t total_time = 0;
  for (const auto& elem : m_timer_map) {
    total_time += elem.second.get_milliseconds();
  }
  if (verbose_op("total")) {
    NGRAPH_HE_LOG(3) << "\033[1;32m"
                     << "Total time " << total_time << " (ms) \033[0m";
  }

  // Send outputs to client.
  if (enable_client()) {
    send_client_results();
  }
  return true;
}

void HESealExecutable::send_client_results() {
  NGRAPH_HE_LOG(3) << "Sending results to client";
  NGRAPH_CHECK(m_client_outputs.size() == 1,
               "HESealExecutable only supports output size 1 (got ",
               get_results().size(), "");

  std::vector<proto::HETensor> proto_tensors;
  m_client_outputs[0]->write_to_protos(proto_tensors);

  for (const auto& proto_tensor : proto_tensors) {
    proto::TCPMessage result_msg;
    result_msg.set_type(proto::TCPMessage_Type_RESPONSE);
    *result_msg.add_he_tensors() = proto_tensor;

    auto result_shape = result_msg.he_tensors(0).shape();
    NGRAPH_HE_LOG(3) << "Server sending result with shape "
                     << Shape{result_shape.begin(), result_shape.end()};
    m_session->write_message(std::move(result_msg));
  }

  // Wait until message is written
  std::unique_lock<std::mutex> mlock(m_result_mutex);
  std::condition_variable& writing_cond = m_session->is_writing_cond();
  writing_cond.wait(mlock, [this] { return !m_session->is_writing(); });
}

void HESealExecutable::generate_calls(
    const element::Type& type, const NodeWrapper& node_wrapper,
    const std::vector<std::shared_ptr<HETensor>>& out,
    const std::vector<std::shared_ptr<HETensor>>& args) {
  const Node& node = *node_wrapper.get_node();
  bool verbose = verbose_op(node);
  std::string node_op = node.description();

// We want to check that every OP_TYPEID enumeration is included in the
// list. These GCC flags enable compile-time checking so that if an
//      enumeration
// is not in the list an error is generated.
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
  switch (node_wrapper.get_typeid()) {
    case OP_TYPEID::Add: {
      add_seal(args[0]->data(), args[1]->data(), out[0]->data(),
               out[0]->get_batched_element_count(), type, m_he_seal_backend);
      break;
    }
    case OP_TYPEID::AvgPool: {
      const auto avg_pool = static_cast<const op::AvgPool*>(&node);
      Shape op_in_shape = args[0]->get_packed_shape();
      Shape op_out_shape = out[0]->get_packed_shape();

      if (verbose) {
        NGRAPH_HE_LOG(3) << "AvgPool " << op_in_shape << " => " << op_out_shape;
      }

      avg_pool_seal(
          args[0]->data(), out[0]->data(), op_in_shape, op_out_shape,
          avg_pool->get_window_shape(), avg_pool->get_window_movement_strides(),
          avg_pool->get_padding_below(), avg_pool->get_padding_above(),
          avg_pool->get_include_padding_in_avg_computation(),
          out[0]->get_batch_size(), m_he_seal_backend);
      rescale_seal(out[0]->data(), m_he_seal_backend, verbose);
      break;
    }
    case OP_TYPEID::BatchNormInference: {
      const auto bn = static_cast<const ngraph::op::BatchNormInference*>(&node);
      double eps = bn->get_eps_value();
      NGRAPH_CHECK(args.size() == 5, "BatchNormInference has ", args.size(),
                   "arguments (expected 5).");

      auto gamma = args[0];
      auto beta = args[1];
      auto input = args[2];
      auto mean = args[3];
      auto variance = args[4];

      batch_norm_inference_seal(eps, gamma->data(), beta->data(), input->data(),
                                mean->data(), variance->data(), out[0]->data(),
                                args[2]->get_packed_shape(), m_batch_size,
                                m_he_seal_backend);
      break;
    }
    case OP_TYPEID::BoundedRelu: {
      const auto bounded_relu = static_cast<const op::BoundedRelu*>(&node);
      float alpha = bounded_relu->get_alpha();
      size_t output_size = args[0]->get_batched_element_count();
      if (enable_client()) {
        handle_server_relu_op(args[0], out[0], node_wrapper);
      } else {
        NGRAPH_WARN << "Performing BoundedRelu without client is not "
                       "privacy-preserving ";
        NGRAPH_CHECK(output_size == args[0]->data().size(), "output size ",
                     output_size, " doesn't match number of elements",
                     out[0]->data().size());
        bounded_relu_seal(args[0]->data(), out[0]->data(), alpha, output_size,
                          m_he_seal_backend);
      }
      break;
    }
    case OP_TYPEID::Broadcast: {
      const auto broadcast = static_cast<const op::Broadcast*>(&node);
      broadcast_seal(args[0]->data(), out[0]->data(),
                     args[0]->get_packed_shape(), out[0]->get_packed_shape(),
                     broadcast->get_broadcast_axes());
      break;
    }
    case OP_TYPEID::BroadcastLike:
      break;
    case OP_TYPEID::Concat: {
      const auto* concat = static_cast<const op::Concat*>(&node);
      std::vector<Shape> in_shapes;
      std::vector<std::vector<HEType>> in_args;
      for (auto& arg : args) {
        in_args.push_back(arg->data());
        in_shapes.push_back(arg->get_packed_shape());
      }
      concat_seal(in_args, out[0]->data(), in_shapes,
                  out[0]->get_packed_shape(), concat->get_concatenation_axis());
      break;
    }
    case OP_TYPEID::Constant: {
      const auto* constant = static_cast<const op::Constant*>(&node);
      constant_seal(out[0]->data(), type, constant->get_data_ptr(),
                    m_he_seal_backend, out[0]->get_batched_element_count());
      break;
    }
    case OP_TYPEID::Convolution: {
      const auto* c = static_cast<const op::Convolution*>(&node);
      const auto& window_movement_strides = c->get_window_movement_strides();
      const auto& window_dilation_strides = c->get_window_dilation_strides();
      const auto& padding_below = c->get_padding_below();
      const auto& padding_above = c->get_padding_above();
      const auto& data_dilation_strides = c->get_data_dilation_strides();

      Shape in_shape0 = args[0]->get_packed_shape();
      Shape in_shape1 = args[1]->get_packed_shape();

      if (verbose) {
        NGRAPH_HE_LOG(3) << in_shape0 << " Conv " << in_shape1 << " => "
                         << out[0]->get_packed_shape();
      }
      convolution_seal(args[0]->data(), args[1]->data(), out[0]->data(),
                       in_shape0, in_shape1, out[0]->get_packed_shape(),
                       window_movement_strides, window_dilation_strides,
                       padding_below, padding_above, data_dilation_strides, 0,
                       1, 1, 0, 0, 1, false, type, m_batch_size,
                       m_he_seal_backend, verbose);

      rescale_seal(out[0]->data(), m_he_seal_backend, verbose);

      break;
    }
    case OP_TYPEID::Divide: {
      Shape in_shape0 = args[0]->get_packed_shape();
      Shape in_shape1 = args[1]->get_packed_shape();

      divide_seal(args[0]->data(), args[1]->data(), out[0]->data(),
                  out[0]->get_batched_element_count(), type, m_he_seal_backend);
      break;
    }
    case OP_TYPEID::Dot: {
      const auto* dot = static_cast<const op::Dot*>(&node);

      Shape in_shape0 = args[0]->get_packed_shape();
      Shape in_shape1 = args[1]->get_packed_shape();

      if (verbose) {
        NGRAPH_HE_LOG(3) << in_shape0 << " dot " << in_shape1;
      }
      dot_seal(args[0]->data(), args[1]->data(), out[0]->data(), in_shape0,
               in_shape1, out[0]->get_packed_shape(),
               dot->get_reduction_axes_count(), type, m_he_seal_backend);
      rescale_seal(out[0]->data(), m_he_seal_backend, verbose);

      break;
    }
    case OP_TYPEID::Exp: {
      if (enable_client()) {
        NGRAPH_CHECK(false, "Exp not implemented for client-aided model ");
      } else {
        NGRAPH_WARN
            << " Performing Exp without client is not privacy-preserving ";
        exp_seal(args[0]->data(), out[0]->data(),
                 args[0]->get_batched_element_count(), m_he_seal_backend);
      }
      break;
    }
    case OP_TYPEID::Max: {
      const auto* max = static_cast<const op::Max*>(&node);
      auto reduction_axes = max->get_reduction_axes();
      NGRAPH_CHECK(!args[0]->is_packed() ||
                       (reduction_axes.find(0) == reduction_axes.end()),
                   "Max reduction axes cannot contain 0 for packed tensors");
      if (enable_client()) {
        NGRAPH_CHECK(false, "Max not implemented for client-aided model");
      } else {
        NGRAPH_WARN << "Performing Max without client is not "
                       "privacy-preserving";
        size_t output_size = args[0]->get_batched_element_count();
        NGRAPH_CHECK(output_size == args[0]->data().size(), "output size ",
                     output_size, " doesn't match number of elements",
                     out[0]->data().size());
        max_seal(args[0]->data(), out[0]->data(), args[0]->get_packed_shape(),
                 out[0]->get_packed_shape(), max->get_reduction_axes(),
                 out[0]->get_batch_size(), m_he_seal_backend);
      }
      break;
    }
    case OP_TYPEID::MaxPool: {
      const auto* max_pool = static_cast<const op::MaxPool*>(&node);
      if (enable_client()) {
        handle_server_max_pool_op(args[0], out[0], node_wrapper);
      } else {
        NGRAPH_WARN << "Performing MaxPool without client is not "
                       "privacy-preserving";
        size_t output_size = args[0]->get_batched_element_count();
        NGRAPH_CHECK(output_size == args[0]->data().size(), "output size ",
                     output_size, " doesn't match number of elements",
                     out[0]->data().size());
        max_pool_seal(args[0]->data(), out[0]->data(),
                      args[0]->get_packed_shape(), out[0]->get_packed_shape(),
                      max_pool->get_window_shape(),
                      max_pool->get_window_movement_strides(),
                      max_pool->get_padding_below(),
                      max_pool->get_padding_above(), m_he_seal_backend);
      }
      break;
    }
    case OP_TYPEID::Minimum: {
      minimum_seal(args[0]->data(), args[1]->data(), out[0]->data(),
                   out[0]->get_batched_element_count(), m_he_seal_backend);
      break;
    }
    case OP_TYPEID::Multiply: {
      multiply_seal(args[0]->data(), args[1]->data(), out[0]->data(),
                    out[0]->get_batched_element_count(), type,
                    m_he_seal_backend);
      rescale_seal(out[0]->data(), m_he_seal_backend, verbose);
      break;
    }
    case OP_TYPEID::Negative: {
      negate_seal(args[0]->data(), out[0]->data(),
                  out[0]->get_batched_element_count(), type, m_he_seal_backend);
      break;
    }
    case OP_TYPEID::Pad: {
      const auto* pad = static_cast<const op::Pad*>(&node);
      pad_seal(args[0]->data(), args[1]->data(), out[0]->data(),
               args[0]->get_packed_shape(), out[0]->get_packed_shape(),
               pad->get_padding_below(), pad->get_padding_above(),
               pad->get_pad_mode());
      break;
    }
    case OP_TYPEID::Parameter: {
      NGRAPH_HE_LOG(3) << "Skipping parameter";
      break;
    }
    case OP_TYPEID::Passthrough: {
      const auto* passthrough = static_cast<const op::Passthrough*>(&node);
      throw unsupported_op{"Unsupported operation language: " +
                           passthrough->language()};
    }
    case OP_TYPEID::Power: {
      // TODO(fboemer): implement with client
      NGRAPH_WARN
          << "Performing Power without client is not privacy preserving ";

      power_seal(args[0]->data(), args[1]->data(), out[0]->data(),
                 out[0]->data().size(), type, m_he_seal_backend);
      break;
    }
    case OP_TYPEID::Relu: {
      if (enable_client()) {
        handle_server_relu_op(args[0], out[0], node_wrapper);
      } else {
        NGRAPH_WARN
            << "Performing Relu without client is not privacy preserving ";
        size_t output_size = args[0]->get_batched_element_count();
        NGRAPH_CHECK(output_size == args[0]->data().size(), "output size ",
                     output_size, "doesn't match number of elements",
                     out[0]->data().size());
        relu_seal(args[0]->data(), out[0]->data(), output_size,
                  m_he_seal_backend);
      }
      break;
    }
    case OP_TYPEID::Reshape: {
      const auto* reshape = static_cast<const op::Reshape*>(&node);
      if (verbose) {
        NGRAPH_HE_LOG(3) << args[0]->get_packed_shape() << " reshape "
                         << out[0]->get_packed_shape();
      }
      reshape_seal(args[0]->data(), out[0]->data(), args[0]->get_packed_shape(),
                   reshape->get_input_order(), out[0]->get_packed_shape());

      break;
    }
    case OP_TYPEID::Result: {
      result_seal(args[0]->data(), out[0]->data(),
                  out[0]->get_batched_element_count(), m_he_seal_backend);
      break;
    }
    case OP_TYPEID::Reverse: {
      const auto* reverse = static_cast<const op::Reverse*>(&node);
      if (verbose) {
        NGRAPH_HE_LOG(3) << args[0]->get_packed_shape() << " reshape "
                         << out[0]->get_packed_shape();
      }
      reverse_seal(args[0]->data(), out[0]->data(), args[0]->get_packed_shape(),
                   out[0]->get_packed_shape(), reverse->get_reversed_axes());
      break;
    }
    case OP_TYPEID::ScalarConstantLike: {
      break;
    }
    case OP_TYPEID::Slice: {
      const auto* slice = static_cast<const op::Slice*>(&node);
      const Shape& in_shape = args[0]->get_packed_shape();
      const Shape& out_shape = out[0]->get_packed_shape();
      const Coordinate& lower_bounds = slice->get_lower_bounds();
      Coordinate upper_bounds = slice->get_upper_bounds();
      const Strides& strides = slice->get_strides();

      if (verbose) {
        NGRAPH_HE_LOG(3) << "in_shape " << in_shape;
        NGRAPH_HE_LOG(3) << "out_shape " << out_shape;
        NGRAPH_HE_LOG(3) << "lower_bounds " << lower_bounds;
        NGRAPH_HE_LOG(3) << "upper_bounds " << upper_bounds;
        NGRAPH_HE_LOG(3) << "strides " << strides;
      }

      if (!upper_bounds.empty() && !upper_bounds.empty() &&
          (upper_bounds[0] > in_shape[0])) {
        NGRAPH_CHECK(upper_bounds[0] == out[0]->get_batch_size(),
                     "Slice upper bound shape ", upper_bounds,
                     " is not compatible with tensor output shape ",
                     out[0]->get_shape());
        upper_bounds[0] = 1;
        if (verbose) {
          NGRAPH_HE_LOG(3) << "new upper_bounds " << upper_bounds;
        }
      }

      slice_seal(args[0]->data(), out[0]->data(), in_shape, lower_bounds,
                 upper_bounds, strides, out_shape);

      break;
    }
    case OP_TYPEID::Softmax: {
      const auto* softmax = static_cast<const op::Softmax*>(&node);
      auto axes = softmax->get_axes();
      NGRAPH_CHECK(!args[0]->is_packed() || (axes.find(0) == axes.end()),
                   "Softmax axes cannot contain 0 for packed tensors");

      softmax_seal(args[0]->data(), out[0]->data(), args[0]->get_packed_shape(),
                   softmax->get_axes(), type, m_he_seal_backend);
      break;
    }
    case OP_TYPEID::Subtract: {
      subtract_seal(args[0]->data(), args[1]->data(), out[0]->data(),
                    out[0]->get_batched_element_count(), type,
                    m_he_seal_backend);
      break;
    }
    case OP_TYPEID::Sum: {
      const auto* sum = static_cast<const op::Sum*>(&node);
      sum_seal(args[0]->data(), out[0]->data(), args[0]->get_packed_shape(),
               out[0]->get_packed_shape(), sum->get_reduction_axes(), type,
               m_he_seal_backend);
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
    case OP_TYPEID::DynBroadcast:
    case OP_TYPEID::DynPad:
    case OP_TYPEID::DynReshape:
    case OP_TYPEID::DynSlice:
    case OP_TYPEID::DynReplaceSlice:
    case OP_TYPEID::EmbeddingLookup:
    case OP_TYPEID::Equal:
    case OP_TYPEID::Erf:
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
    case OP_TYPEID::Maximum:
    case OP_TYPEID::MaxPoolBackprop:
    case OP_TYPEID::Min:
    case OP_TYPEID::Not:
    case OP_TYPEID::NotEqual:
    case OP_TYPEID::OneHot:
    case OP_TYPEID::Or:
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
}  // namespace he

void HESealExecutable::handle_server_max_pool_op(
    const std::shared_ptr<HETensor>& arg, const std::shared_ptr<HETensor>& out,
    const NodeWrapper& node_wrapper) {
  NGRAPH_HE_LOG(3) << "Server handle_server_max_pool_op";

  const Node& node = *node_wrapper.get_node();
  bool verbose = verbose_op(node);
  const auto* max_pool = static_cast<const op::MaxPool*>(&node);

  m_max_pool_done = false;

  Shape unpacked_arg_shape = node.get_input_shape(0);
  Shape out_shape = HETensor::pack_shape(node.get_output_shape(0));

  // TODO(fboemer): call max_pool_seal directly?
  std::vector<std::vector<size_t>> maximize_lists = max_pool_seal_max_list(
      unpacked_arg_shape, out_shape, max_pool->get_window_shape(),
      max_pool->get_window_movement_strides(), max_pool->get_padding_below(),
      max_pool->get_padding_above());

  m_max_pool_data.clear();

  for (const auto& maximize_list : maximize_lists) {
    proto::TCPMessage proto_msg;
    proto_msg.set_type(proto::TCPMessage_Type_REQUEST);

    json js = {{"function", node.description()}};
    proto::Function f;
    f.set_function(js.dump());
    *proto_msg.mutable_function() = f;

    std::vector<HEType> cipher_batch;
    cipher_batch.reserve(maximize_list.size());
    for (const size_t max_ind : maximize_list) {
      cipher_batch.emplace_back(arg->data(max_ind));
    }

    NGRAPH_CHECK(!cipher_batch.empty(), "Maxpool cipher batch is empty");

    HETensor max_pool_tensor(
        arg->get_element_type(),
        Shape{cipher_batch[0].batch_size(), cipher_batch.size()},
        cipher_batch[0].plaintext_packing(), cipher_batch[0].complex_packing(),
        true, m_he_seal_backend);
    max_pool_tensor.data() = cipher_batch;
    std::vector<proto::HETensor> proto_tensors;
    max_pool_tensor.write_to_protos(proto_tensors);
    NGRAPH_CHECK(proto_tensors.size() == 1,
                 "Only support MaxPool with 1 proto tensor");
    *proto_msg.add_he_tensors() = proto_tensors[0];

    // Send list of ciphertexts to maximize over to client
    if (verbose) {
      NGRAPH_HE_LOG(3) << "Sending " << cipher_batch.size()
                       << " Maxpool ciphertexts to client";
    }

    TCPMessage max_pool_message(std::move(proto_msg));
    m_session->write_message(std::move(max_pool_message));

    // Acquire lock
    std::unique_lock<std::mutex> mlock(m_max_pool_mutex);

    // Wait until max is done
    m_max_pool_cond.wait(mlock,
                         std::bind(&HESealExecutable::max_pool_done, this));

    // Reset for next max_pool call
    m_max_pool_done = false;
  }
  out->data() = m_max_pool_data;
}

void HESealExecutable::handle_server_relu_op(
    const std::shared_ptr<HETensor>& arg, const std::shared_ptr<HETensor>& out,
    const NodeWrapper& node_wrapper) {
  NGRAPH_HE_LOG(3) << "Server handle_server_relu_op"
                   << (enable_garbled_circuits() ? " with garbled circuits"
                                                 : "");

  auto type_id = node_wrapper.get_typeid();
  NGRAPH_CHECK(type_id == OP_TYPEID::Relu || type_id == OP_TYPEID::BoundedRelu,
               "only support relu / bounded relu");

  const Node& node = *node_wrapper.get_node();
  bool verbose = verbose_op(node);
  size_t element_count = arg->data().size();

  size_t smallest_ind =
      match_to_smallest_chain_index(arg->data(), m_he_seal_backend);
  if (verbose) {
    NGRAPH_HE_LOG(3) << "Matched moduli to chain ind " << smallest_ind;
  }

  m_relu_data.resize(element_count, HEType(HEPlaintext(), false));

  // TODO(fboemer): tune
  const size_t max_relu_message_cnt = 1000;

  m_unknown_relu_idx.clear();
  m_unknown_relu_idx.reserve(element_count);

  // Process known values
  for (size_t relu_idx = 0; relu_idx < element_count; ++relu_idx) {
    auto& he_type = arg->data(relu_idx);
    if (he_type.is_plaintext()) {
      m_relu_data[relu_idx].set_plaintext(HEPlaintext());
      if (type_id == OP_TYPEID::Relu) {
        scalar_relu_seal(he_type.get_plaintext(),
                         m_relu_data[relu_idx].get_plaintext());
      } else {
        const auto* bounded_relu = static_cast<const op::BoundedRelu*>(&node);
        float alpha = bounded_relu->get_alpha();
        scalar_bounded_relu_seal(he_type.get_plaintext(),
                                 m_relu_data[relu_idx].get_plaintext(), alpha);
      }
    } else {
      m_unknown_relu_idx.emplace_back(relu_idx);
    }
  }
  auto process_unknown_relu_ciphers_batch =
      [&](std::vector<HEType>& cipher_batch) {
        if (verbose) {
          NGRAPH_HE_LOG(3) << "Sending relu request size "
                           << cipher_batch.size();
        }

        he_proto::TCPMessage proto_msg;
        proto_msg.set_type(he_proto::TCPMessage_Type_REQUEST);
        *proto_msg.mutable_function() = node_to_proto_function(
            node_wrapper,
            {{"enable_gc", he::bool_to_string(enable_garbled_circuits())}});
        std::string function_str = proto_msg.function().function();

        // TODO: set complex_packing to correct values?
        auto relu_tensor = std::make_shared<HETensor>(
            arg->get_element_type(),
            Shape{cipher_batch[0].batch_size(), cipher_batch.size()},
            arg->is_packed(), false, true, m_he_seal_backend);
        relu_tensor->data() = cipher_batch;
        NGRAPH_INFO << "relu tensor shape " << relu_tensor->get_shape()
                    << " with batch size " << relu_tensor->get_batch_size();

        if (enable_garbled_circuits()) {
          // Masks input values
          m_aby_executor->prepare_aby_circuit(function_str, relu_tensor);
        }

        std::vector<proto::HETensor> proto_tensors;
        relu_tensor.write_to_protos(proto_tensors);
        for (const auto& proto_tensor : proto_tensors) {
          proto::TCPMessage proto_msg;
          proto_msg.set_type(proto::TCPMessage_Type_REQUEST);

          // TODO(fboemer): factor out serializing the function
          json js = {{"function", node.description()}};
          if (type_id == OP_TYPEID::BoundedRelu) {
            const auto* bounded_relu =
                static_cast<const op::BoundedRelu*>(&node);
            float alpha = bounded_relu->get_alpha();
            js["bound"] = alpha;
          }

          proto::Function f;
          f.set_function(js.dump());
          *proto_msg.mutable_function() = f;

          *proto_msg.add_he_tensors() = proto_tensor;
          TCPMessage relu_message(std::move(proto_msg));

          NGRAPH_HE_LOG(5) << "Server writing relu request message";
          m_session->write_message(std::move(relu_message));

          if (enable_garbled_circuits()) {
            m_aby_executor->run_aby_circuit(function_str, relu_tensor);
            NGRAPH_INFO << "Server done running relu circuit";
          }
        };

        // Process unknown values
        std::vector<HEType> relu_ciphers_batch;
        relu_ciphers_batch.reserve(max_relu_message_cnt);

        for (const auto& unknown_relu_idx : m_unknown_relu_idx) {
          NGRAPH_CHECK(arg->data(unknown_relu_idx).is_ciphertext(),
                       "HEType should be ciphertext");
          relu_ciphers_batch.emplace_back(arg->data(unknown_relu_idx));
          if (relu_ciphers_batch.size() == max_relu_message_cnt) {
            process_unknown_relu_ciphers_batch(relu_ciphers_batch);
            relu_ciphers_batch.clear();
          }
        }
        if (!relu_ciphers_batch.empty()) {
          process_unknown_relu_ciphers_batch(relu_ciphers_batch);
          relu_ciphers_batch.clear();
        }

        // Wait until all batches have been processed
        std::unique_lock<std::mutex> mlock(m_relu_mutex);
        m_relu_cond.wait(mlock, [=]() {
          return m_relu_done_count == m_unknown_relu_idx.size();
        });
        m_relu_done_count = 0;

        out->data() = m_relu_data;
      }

}  // namespace he
}  // namespace he
