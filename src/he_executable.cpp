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

#include "he_cipher_tensor.hpp"
#include "he_executable.hpp"
#include "he_plain_tensor.hpp"
#include "he_tensor.hpp"
#include "kernel/add.hpp"
#include "kernel/avg_pool.hpp"
#include "kernel/batch_norm_inference.hpp"
#include "kernel/broadcast.hpp"
#include "kernel/concat.hpp"
#include "kernel/constant.hpp"
#include "kernel/convolution.hpp"
#include "kernel/dot.hpp"
#include "kernel/max_pool.hpp"
#include "kernel/multiply.hpp"
#include "kernel/negate.hpp"
#include "kernel/pad.hpp"
#include "kernel/reshape.hpp"
#include "kernel/result.hpp"
#include "kernel/reverse.hpp"
#include "kernel/slice.hpp"
#include "kernel/subtract.hpp"
#include "kernel/sum.hpp"
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
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/util.hpp"
#include "seal/ckks/seal_ckks_util.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_util.hpp"

using namespace std;
using namespace ngraph;
using descriptor::layout::DenseTensorLayout;

runtime::he::HEExecutable::HEExecutable(const shared_ptr<Function>& function,
                                        bool enable_performance_collection,
                                        const HEBackend* he_backend,
                                        bool encrypt_data, bool encrypt_model,
                                        bool batch_data)
    : m_he_backend(he_backend),
      m_encrypt_data(encrypt_data),
      m_encrypt_model(encrypt_model),
      m_batch_data(batch_data),
      m_enable_client(std::getenv("NGRAPH_ENABLE_CLIENT") != nullptr),
      m_batch_size(1),
      m_port(34000),
      m_relu_done(false),
      m_session_started(false),
      m_client_inputs_received(false) {
  NGRAPH_ASSERT(he_backend != nullptr) << "he_backend == nullptr";
  // TODO: move get_context to HEBackend
  auto he_seal_backend = (runtime::he::he_seal::HESealBackend*)he_backend;
  NGRAPH_ASSERT(he_seal_backend != nullptr) << "he_seal_backend == nullptr";
  m_context = he_seal_backend->get_context();

  m_is_compiled = true;
  pass::Manager pass_manager;
  pass_manager.register_pass<pass::LikeReplacement>();
  pass_manager.register_pass<pass::AssignLayout<DenseTensorLayout>>();
  pass_manager.register_pass<pass::Liveness>();
  pass_manager.run_passes(function);

  for (const shared_ptr<Node>& node : function->get_ordered_ops()) {
    m_wrapped_nodes.emplace_back(node);
  }
  set_parameters_and_results(*function);

  if (m_enable_client) {
    NGRAPH_INFO << "Enable client";

    // only support parameter size 1 for now
    NGRAPH_ASSERT(get_parameters().size() == 1)
        << "HEExecutable only supports parameter size 1 (got "
        << get_parameters().size() << ")";
    // only support function output size 1 for now
    NGRAPH_ASSERT(get_results().size() == 1)
        << "HEExecutable only supports output size 1 (got "
        << get_results().size() << "";

    // Start server
    NGRAPH_INFO << "Starting server";
    start_server();

    // Send encryption parameters
    stringstream param_stream;
    shared_ptr<HEEncryptionParameters> parms =
        he_backend->get_encryption_parameters();
    NGRAPH_ASSERT(parms != nullptr) << "HEEncryptionParameters == nullptr";

    // only support parameter size 1 for now
    NGRAPH_ASSERT(get_parameters().size() == 1)
        << "HEExecutable only supports parameter size 1 (got "
        << get_parameters().size() << ")";

    const Shape& shape = (get_parameters()[0])->get_shape();
    if (m_batch_data) {
      m_batch_size = shape[0];
      NGRAPH_INFO << "Setting batch_size " << m_batch_size;
    }

    parms->save(param_stream);
    auto parms_message =
        TCPMessage(MessageType::encryption_parameters, 1, param_stream);

    unique_lock<mutex> mlock(m_session_mutex);
    NGRAPH_INFO << "Waiting until client is connected";
    m_session_cond.wait(mlock, std::bind(&HEExecutable::session_started, this));
    NGRAPH_INFO << "Session started";

    m_session->do_write(parms_message);
  }
}

void runtime::he::HEExecutable::accept_connection() {
  NGRAPH_INFO << "Server accepting connections";
  auto server_callback =
      bind(&runtime::he::HEExecutable::handle_message, this, placeholders::_1);

  m_acceptor->async_accept([this, server_callback](boost::system::error_code ec,
                                                   tcp::socket socket) {
    if (!ec) {
      NGRAPH_INFO << "Connection accepted";
      m_session = make_shared<TCPSession>(move(socket), server_callback);
      m_session->start();

      std::lock_guard<mutex> guard(m_session_mutex);
      m_session_started = true;
      m_session_cond.notify_all();
    } else {
      NGRAPH_INFO << "error " << ec.message();
      // accept_connection();
    }
  });
}

void runtime::he::HEExecutable::start_server() {
  tcp::resolver resolver(m_io_context);
  tcp::endpoint server_endpoints(tcp::v4(), m_port);
  m_acceptor = make_shared<tcp::acceptor>(m_io_context, server_endpoints);

  accept_connection();
  m_thread = thread([this]() { m_io_context.run(); });
}

void runtime::he::HEExecutable::handle_message(
    const runtime::he::TCPMessage& message) {
  MessageType msg_type = message.message_type();

  // NGRAPH_INFO << "Server received message type: "
  //            << message_type_to_string(msg_type);

  if (msg_type == MessageType::execute) {
    // Get Ciphertexts from message
    size_t count = message.count();
    vector<seal::Ciphertext> ciphertexts;
    size_t ciphertext_size = message.element_size();

    assert(m_context != nullptr);
    print_seal_context(*m_context);

    NGRAPH_INFO << "Loading " << count << " ciphertexts";
    for (size_t i = 0; i < count; ++i) {
      stringstream stream;
      stream.write(message.data_ptr() + i * ciphertext_size, ciphertext_size);
      seal::Ciphertext c;
      c.load(m_context, stream);
      ciphertexts.emplace_back(c);
    }
    NGRAPH_INFO << "Done loading " << count << " ciphertexts";

    vector<shared_ptr<runtime::he::HECiphertext>> he_cipher_inputs;
    for (const auto cipher : ciphertexts) {
      auto wrapper =
          make_shared<runtime::he::he_seal::SealCiphertextWrapper>(cipher);
      he_cipher_inputs.emplace_back(wrapper);
    }

    // only support parameter size 1 for now
    NGRAPH_ASSERT(get_parameters().size() == 1)
        << "HEExecutable only supports parameter size 1 (got "
        << get_parameters().size() << ")";

    // only support function output size 1 for now
    NGRAPH_ASSERT(get_results().size() == 1)
        << "HEExecutable only supports output size 1 (got "
        << get_results().size() << "";

    // Load function with parameters
    size_t num_param_elements = 0;
    const ParameterVector& input_parameters = get_parameters();
    for (auto input_param : input_parameters) {
      num_param_elements += shape_size(input_param->get_shape());
    }
    num_param_elements /= m_batch_size;
    NGRAPH_ASSERT(count == num_param_elements)
        << "Count " << count
        << " does not match number of parameter elements ( "
        << num_param_elements << ")";

    NGRAPH_INFO << "Setting m_client_inputs";
    size_t parameter_size_index = 0;
    for (auto input_param : input_parameters) {
      const auto& shape = input_param->get_shape();
      size_t param_size = shape_size(shape) / m_batch_size;
      NGRAPH_INFO << "shape " << join(shape, "x");
      NGRAPH_INFO << "param_size " << param_size;
      NGRAPH_INFO << "m_batch_data? " << m_batch_data;

      auto element_type = input_param->get_element_type();
      auto input_tensor = dynamic_pointer_cast<HECipherTensor>(
          m_he_backend->create_cipher_tensor(
              element_type, input_param->get_shape(), m_batch_data));

      vector<shared_ptr<runtime::he::HECiphertext>> cipher_elements{
          he_cipher_inputs.begin() + parameter_size_index,
          he_cipher_inputs.begin() + parameter_size_index + param_size};

      NGRAPH_ASSERT(cipher_elements.size() == param_size)
          << "Incorrect number of elements for parameter";

      input_tensor->set_elements(cipher_elements);
      m_client_inputs.emplace_back(input_tensor);
      parameter_size_index += param_size;
    }

    NGRAPH_ASSERT(m_client_inputs.size() == get_parameters().size())
        << "Client inputs size " << m_client_inputs.size() << "; expected "
        << get_parameters().size();

    std::lock_guard<mutex> guard(m_client_inputs_mutex);
    m_client_inputs_received = true;
    m_client_inputs_cond.notify_all();

  } else if (msg_type == MessageType::public_key) {
    seal::PublicKey key;
    std::stringstream key_stream;
    key_stream.write(message.data_ptr(), message.element_size());
    key.load(m_context, key_stream);

    // TODO: move set_public_key to HEBackend
    auto he_seal_backend = (runtime::he::he_seal::HESealBackend*)m_he_backend;
    he_seal_backend->set_public_key(key);

    NGRAPH_INFO << "Server set public key";

  } else if (msg_type == MessageType::eval_key) {
    seal::RelinKeys keys;
    std::stringstream key_stream;
    key_stream.write(message.data_ptr(), message.element_size());
    keys.load(m_context, key_stream);

    // TODO: move set_relin_keys to HEBackend
    auto he_seal_backend = (runtime::he::he_seal::HESealBackend*)m_he_backend;
    he_seal_backend->set_relin_keys(keys);

    // Send inference parameter shape
    const ParameterVector& input_parameters = get_parameters();
    size_t num_param_elements = 0;
    for (const auto& param : input_parameters) {
      auto& shape = param->get_shape();
      num_param_elements += shape_size(shape);
      NGRAPH_INFO << "Parameter shape " << join(shape, "x");
    }

    if (m_batch_data) {
      NGRAPH_INFO << "num_param_elements before batch size divide "
                  << num_param_elements;
      num_param_elements /= m_batch_size;
      NGRAPH_INFO << "num_param_elements after batch size divide "
                  << num_param_elements;
    }

    NGRAPH_INFO << "Requesting total of " << num_param_elements
                << " parameter elements";
    runtime::he::TCPMessage parameter_message{MessageType::parameter_size, 1,
                                              sizeof(num_param_elements),
                                              (char*)&num_param_elements};

    NGRAPH_INFO << "Server sending message of type: parameter_size";
    m_session->do_write(parameter_message);
  } else if (msg_type == MessageType::relu_result) {
    std::lock_guard<mutex> guard(m_relu_mutex);

    size_t element_count = message.count();
    size_t element_size = message.element_size();
    m_relu_ciphertexts.clear();

    NGRAPH_INFO << "Got " << element_count << " ciphertexts";
    NGRAPH_INFO << "element_size " << element_size;

    for (size_t element_idx = 0; element_idx < element_count; ++element_idx) {
      seal::Ciphertext cipher;
      stringstream cipher_stream;
      cipher_stream.write(message.data_ptr() + element_idx * element_size,
                          element_size);
      cipher.load(m_context, cipher_stream);

      auto wrapper =
          make_shared<runtime::he::he_seal::SealCiphertextWrapper>(cipher);
      auto he_ciphertext =
          dynamic_pointer_cast<runtime::he::HECiphertext>(wrapper);

      NGRAPH_ASSERT(he_ciphertext != nullptr)
          << "HECiphertext is not SealPlaintextWrapper";

      m_relu_ciphertexts.emplace_back(he_ciphertext);
    }
    NGRAPH_INFO << "Done loading Relu ciphertexts";

    // Notify condition variable
    m_relu_done = true;
    m_relu_cond.notify_all();
  } else if (msg_type == MessageType::max_result) {
    std::lock_guard<mutex> guard(m_max_mutex);

    size_t element_count = message.count();
    size_t element_size = message.element_size();

    for (size_t element_idx = 0; element_idx < element_count; ++element_idx) {
      seal::Ciphertext cipher;
      stringstream cipher_stream;
      cipher_stream.write(message.data_ptr() + element_idx * element_size,
                          element_size);
      cipher.load(m_context, cipher_stream);

      auto wrapper =
          make_shared<runtime::he::he_seal::SealCiphertextWrapper>(cipher);
      auto he_ciphertext =
          dynamic_pointer_cast<runtime::he::HECiphertext>(wrapper);

      NGRAPH_ASSERT(he_ciphertext != nullptr)
          << "HECiphertext is not SealPlaintextWrapper";

      m_max_ciphertexts.emplace_back(he_ciphertext);
    }
    // Notify condition variable
    m_max_done = true;
    m_max_cond.notify_all();
  } else if (msg_type == MessageType::minimum_result) {
    std::lock_guard<mutex> guard(m_minimum_mutex);

    size_t element_count = message.count();
    size_t element_size = message.element_size();

    for (size_t element_idx = 0; element_idx < element_count; ++element_idx) {
      seal::Ciphertext cipher;
      stringstream cipher_stream;
      cipher_stream.write(message.data_ptr() + element_idx * element_size,
                          element_size);
      cipher.load(m_context, cipher_stream);

      auto wrapper =
          make_shared<runtime::he::he_seal::SealCiphertextWrapper>(cipher);
      auto he_ciphertext =
          dynamic_pointer_cast<runtime::he::HECiphertext>(wrapper);

      NGRAPH_ASSERT(he_ciphertext != nullptr)
          << "HECiphertext is not SealPlaintextWrapper";

      m_minimum_ciphertexts.emplace_back(he_ciphertext);
    }
    // Notify condition variable
    m_minimum_done = true;
    m_minimum_cond.notify_all();
  } else {
    stringstream ss;
    ss << "Unsupported message type in server:  "
       << message_type_to_string(msg_type);
    throw ngraph_error(ss.str());
  }
}

vector<runtime::PerformanceCounter>
runtime::he::HEExecutable::get_performance_data() const {
  vector<runtime::PerformanceCounter> rc;
  for (const pair<const Node*, stopwatch> p : m_timer_map) {
    rc.emplace_back(p.first->get_name().c_str(),
                    p.second.get_total_microseconds(),
                    p.second.get_call_count());
  }
  return rc;
}

void runtime::he::HEExecutable::he_validate(
    const vector<shared_ptr<runtime::he::HETensor>>& outputs,
    const vector<shared_ptr<runtime::he::HETensor>>& inputs) {
  const ParameterVector& parameters = get_parameters();
  const ResultVector& results = get_results();
  if (parameters.size() != inputs.size()) {
    stringstream ss;
    ss << "Call input count " << inputs.size()
       << " does not match Function's Parameter count " << parameters.size();
    throw runtime_error(ss.str());
  }
  if (results.size() != outputs.size()) {
    stringstream ss;
    ss << "Call output count " << outputs.size()
       << " does not match Function's Result count " << results.size();
    throw runtime_error(ss.str());
  }

  for (size_t i = 0; i < parameters.size(); i++) {
    if (parameters[i]->get_element_type() != inputs[i]->get_element_type()) {
      stringstream ss;
      ss << "Input " << i << " type '" << inputs[i]->get_element_type()
         << "' does not match Parameter type '"
         << parameters[i]->get_element_type() << "'";
      throw runtime_error(ss.str());
    }
    if (inputs[i]->get_expanded_shape() != parameters[i]->get_shape()) {
      stringstream ss;
      ss << "Input " << i << " shape {" << join(inputs[i]->get_expanded_shape())
         << "} does not match Parameter shape {"
         << join(parameters[i]->get_shape()) << "}";
      throw runtime_error(ss.str());
    }
  }

  for (size_t i = 0; i < results.size(); i++) {
    if (results[i]->get_element_type() != outputs[i]->get_element_type()) {
      stringstream ss;
      ss << "Output " << i << " type '" << outputs[i]->get_element_type()
         << "' does not match Result type '" << results[i]->get_element_type()
         << "'";
      throw runtime_error(ss.str());
    }
    if (results[i]->get_shape() != outputs[i]->get_expanded_shape()) {
      stringstream ss;
      ss << "Output " << i << " shape {"
         << join(outputs[i]->get_expanded_shape())
         << "} does not match Result shape {" << join(results[i]->get_shape())
         << "}";
      throw runtime_error(ss.str());
    }
  }
}

bool runtime::he::HEExecutable::call(
    const vector<shared_ptr<runtime::Tensor>>& outputs,
    const vector<shared_ptr<runtime::Tensor>>& server_inputs) {
  if (m_enable_client) {
    NGRAPH_INFO << "Waiting until m_client_inputs.size() "
                << m_client_inputs.size() << " == " << server_inputs.size();

    unique_lock<mutex> mlock(m_client_inputs_mutex);
    m_client_inputs_cond.wait(
        mlock, std::bind(&HEExecutable::client_inputs_received, this));
    NGRAPH_INFO << "client_inputs_received";

    NGRAPH_ASSERT(m_client_inputs.size() == server_inputs.size())
        << "Recieved incorrect number of inputs from client (got "
        << m_client_inputs.size() << ", expectd " << server_inputs.size();

    NGRAPH_INFO << "Done waiting for m_client_inputs";
  }

  if (m_encrypt_data) {
    NGRAPH_INFO << "Encrypting data";
  }
  if (m_batch_data) {
    NGRAPH_INFO << "Batching data";
  }
  if (m_encrypt_model) {
    NGRAPH_INFO << "Encrypting model";
  }

  // convert inputs to HETensor
  vector<shared_ptr<runtime::he::HETensor>> he_inputs;
  if (m_enable_client) {
    NGRAPH_INFO << "Processing client inputs";
    for (auto& tv : m_client_inputs) {
      he_inputs.push_back(static_pointer_cast<runtime::he::HETensor>(tv));
    }
  } else {
    NGRAPH_INFO << "Processing server inputs";
    for (auto& tv : server_inputs) {
      auto he_input = dynamic_pointer_cast<runtime::he::HETensor>(tv);
      NGRAPH_ASSERT(he_input != nullptr) << "server input is not he tensor";
      he_inputs.push_back(he_input);
    }
  }

  // convert outputs to HETensor
  vector<shared_ptr<runtime::he::HETensor>> he_outputs;
  for (auto& tv : outputs) {
    he_outputs.push_back(static_pointer_cast<runtime::he::HETensor>(tv));
  }

  he_validate(he_outputs, he_inputs);

  unordered_map<descriptor::Tensor*, shared_ptr<runtime::he::HETensor>>
      tensor_map;

  // map function params -> HETensor
  size_t input_count = 0;
  for (auto param : get_parameters()) {
    for (size_t i = 0; i < param->get_output_size(); ++i) {
      descriptor::Tensor* tv = param->get_output_tensor_ptr(i).get();

      if (!m_enable_client && m_encrypt_data) {
        NGRAPH_INFO << "Encrypting parameter " << i;
        auto plain_input = dynamic_pointer_cast<runtime::he::HEPlainTensor>(
            he_inputs[input_count]);
        NGRAPH_ASSERT(plain_input != nullptr) << "Input is not plain tensor";
        auto cipher_input = dynamic_pointer_cast<HECipherTensor>(
            m_he_backend->create_cipher_tensor(plain_input->get_element_type(),
                                               plain_input->get_shape(),
                                               m_batch_data));

#pragma omp parallel for
        for (size_t i = 0; i < plain_input->get_batched_element_count(); ++i) {
          m_he_backend->encrypt(cipher_input->get_element(i),
                                plain_input->get_element(i).get());
        }
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
    if (!dynamic_pointer_cast<op::Result>(output)) {
      throw ngraph_error("One of function's outputs isn't op::Result");
    }
    descriptor::Tensor* tv = output->get_output_tensor_ptr(0).get();
    tensor_map.insert({tv, he_outputs[output_count++]});
  }

  // for each ordered op in the graph
  for (const NodeWrapper& wrapped : m_wrapped_nodes) {
    const Node* op = &wrapped.get_node();
    auto type_id = wrapped.get_typeid();

    NGRAPH_INFO << "\033[1;32m"
                << "[ " << op->get_name() << " ]"
                << "\033[0m";

    if (type_id == OP_TYPEID::Parameter) {
      NGRAPH_INFO << "Parameter shape {" << join(op->get_shape()) << "}";
      continue;
    }

    if (type_id == OP_TYPEID::Constant) {
      NGRAPH_INFO << "Constant shape {" << join(op->get_shape()) << "}";
    }

    // get op inputs from map
    vector<shared_ptr<runtime::he::HETensor>> op_inputs;
    for (const descriptor::Input& input : op->get_inputs()) {
      descriptor::Tensor* tv = input.get_output().get_tensor_ptr().get();
      op_inputs.push_back(tensor_map.at(tv));
    }

    if (m_enable_client && type_id == OP_TYPEID::Result) {
      // Client outputs remain ciphertexts, so don't perform result op on them
      NGRAPH_INFO << "Setting client outputs";
      m_client_outputs = op_inputs;
    }

    // get op outputs from map or create
    vector<shared_ptr<runtime::he::HETensor>> op_outputs;
    for (size_t i = 0; i < op->get_output_size(); ++i) {
      descriptor::Tensor* tv = op->get_output_tensor_ptr(i).get();
      auto it = tensor_map.find(tv);
      if (it == tensor_map.end()) {
        // The output tensor is not in the tensor map so create a new tensor
        const Shape& shape = op->get_output_shape(i);
        const element::Type& element_type = op->get_output_element_type(i);
        string name = op->get_output_tensor(i).get_name();

        // Plaintext output only if all inputs are plaintext
        bool plain_out = all_of(
            op_inputs.begin(), op_inputs.end(),
            [](shared_ptr<runtime::he::HETensor> op_input) {
              return dynamic_pointer_cast<HEPlainTensor>(op_input) != nullptr;
            });
        if (op->is_constant()) {
          plain_out = !m_encrypt_model;
        }

        bool batched_out = any_of(op_inputs.begin(), op_inputs.end(),
                                  [](shared_ptr<runtime::he::HETensor> he_tv) {
                                    return he_tv->is_batched();
                                  });
        if (plain_out) {
          auto otv = make_shared<runtime::he::HEPlainTensor>(
              element_type, shape, m_he_backend,
              m_he_backend->create_empty_plaintext(), batched_out, name);
          tensor_map.insert({tv, otv});
        } else {
          auto otv = make_shared<runtime::he::HECipherTensor>(
              element_type, shape, m_he_backend,
              m_he_backend->create_empty_ciphertext(), batched_out, name);
          tensor_map.insert({tv, otv});
        }
      }
      op_outputs.push_back(tensor_map.at(tv));
    }

    // get op type
    element::Type base_type;
    if (op->get_inputs().empty()) {
      base_type = op->get_element_type();
    } else {
      base_type = op->get_inputs().at(0).get_tensor().get_element_type();
    }

    m_timer_map[op].start();
    generate_calls(base_type, wrapped, op_outputs, op_inputs);
    m_timer_map[op].stop();

    const string op_name = op->description();

    // delete any obsolete tensors
    for (const descriptor::Tensor* t : op->liveness_free_list) {
      for (auto it = tensor_map.begin(); it != tensor_map.end(); ++it) {
        if (it->second->get_name() == t->get_name()) {
          tensor_map.erase(it);
          break;
        }
      }
    }
    NGRAPH_INFO << "\033[1;31m" << op->get_name() << " took "
                << m_timer_map[op].get_milliseconds() << "ms"
                << "\033[0m";
  }
  size_t total_time = 0;
  for (const auto& elem : m_timer_map) {
    total_time += elem.second.get_milliseconds();
  }
  NGRAPH_INFO << "\033[1;32m"
              << "Total time " << total_time << " (ms) \033[0m";

  // Send outputs to client.
  if (m_enable_client) {
    NGRAPH_ASSERT(m_client_outputs.size() == 1)
        << "HEExecutable only supports output size 1 (got "
        << get_results().size() << "";

    std::vector<seal::Ciphertext> seal_output;

    const Shape& output_shape = get_results()[0]->get_shape();
    NGRAPH_INFO << "output shape size " << join(output_shape, "x");
    size_t output_shape_size = shape_size(output_shape) / m_batch_size;

    auto output_cipher_tensor =
        dynamic_pointer_cast<HECipherTensor>(m_client_outputs[0]);

    NGRAPH_ASSERT(output_cipher_tensor != nullptr)
        << "Client outputs are not HECipherTensor";

    std::stringstream cipher_stream;
    output_cipher_tensor->save_elements(cipher_stream);
    m_result_message =
        TCPMessage(MessageType::result, output_shape_size, cipher_stream);

    std::cout << "Writing Result message with " << output_shape_size
              << " ciphertexts " << std::endl;
    m_session->do_write(m_result_message);
  }
  return true;
}

void runtime::he::HEExecutable::generate_calls(
    const element::Type& type, const NodeWrapper& node_wrapper,
    const vector<shared_ptr<HETensor>>& out,
    const vector<shared_ptr<HETensor>>& args) {
  const Node& node = node_wrapper.get_node();
  string node_op = node.description();
  shared_ptr<HECipherTensor> arg0_cipher = nullptr;
  shared_ptr<HEPlainTensor> arg0_plain = nullptr;
  shared_ptr<HECipherTensor> arg1_cipher = nullptr;
  shared_ptr<HEPlainTensor> arg1_plain = nullptr;
  auto out0_cipher = dynamic_pointer_cast<HECipherTensor>(out[0]);
  auto out0_plain = dynamic_pointer_cast<HEPlainTensor>(out[0]);

  if (args.size() > 0) {
    arg0_cipher = dynamic_pointer_cast<HECipherTensor>(args[0]);
    arg0_plain = dynamic_pointer_cast<HEPlainTensor>(args[0]);
    NGRAPH_ASSERT(arg0_cipher == nullptr || arg0_plain == nullptr)
        << "arg0 is netiher cipher nor plain";
    NGRAPH_ASSERT(!(arg0_cipher != nullptr && arg0_plain != nullptr))
        << "arg0 is both cipher and plain?";
  }
  if (args.size() > 1) {
    arg1_cipher = dynamic_pointer_cast<HECipherTensor>(args[1]);
    arg1_plain = dynamic_pointer_cast<HEPlainTensor>(args[1]);
    NGRAPH_ASSERT(arg1_cipher == nullptr || arg1_plain == nullptr)
        << "arg1 is neither cipher nor plain";
    NGRAPH_ASSERT(!(arg1_cipher != nullptr && arg1_plain != nullptr))
        << "arg1 is both cipher and plain?";
  }
  size_t batch_size = 1;
  if (out0_cipher != nullptr) {
    batch_size = out0_cipher->get_batch_size();
  } else if (out0_plain != nullptr) {
    batch_size = out0_plain->get_batch_size();
  }

  stringstream ss;
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
  NGRAPH_INFO << ss.str();
  ss.str("");
  ss << "Outputs: ";
  if (out0_cipher != nullptr) {
    ss << "Cipher";
  } else if (out0_plain != nullptr) {
    ss << "Plain";
  }
  NGRAPH_INFO << ss.str();

  if (batch_size != 1) {
    NGRAPH_INFO << "Batch size " << batch_size;
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
        runtime::he::kernel::add(
            arg0_cipher->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), type, m_he_backend,
            out0_cipher->get_batched_element_count());
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::add(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), type, m_he_backend,
            out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && arg1_cipher != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::add(
            arg0_plain->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), type, m_he_backend,
            out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        runtime::he::kernel::add(arg0_plain->get_elements(),
                                 arg1_plain->get_elements(),
                                 out0_plain->get_elements(), type, m_he_backend,
                                 out0_plain->get_batched_element_count());
      } else {
        throw ngraph_error("Add types not supported.");
      }
      break;
    }
    case OP_TYPEID::AvgPool: {
      const op::AvgPool* avg_pool = static_cast<const op::AvgPool*>(&node);
      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        Shape in_shape = arg0_cipher->get_batched_shape();
        Shape out_shape = out0_cipher->get_batched_shape();
        runtime::he::kernel::avg_pool(
            arg0_cipher->get_elements(), out0_cipher->get_elements(), in_shape,
            out_shape, avg_pool->get_window_shape(),
            avg_pool->get_window_movement_strides(),
            avg_pool->get_padding_below(), avg_pool->get_padding_above(),
            avg_pool->get_include_padding_in_avg_computation(), m_he_backend);

      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        Shape in_shape = arg0_plain->get_batched_shape();
        Shape out_shape = out0_plain->get_batched_shape();
        runtime::he::kernel::avg_pool(
            arg0_plain->get_elements(), out0_plain->get_elements(), in_shape,
            out_shape, avg_pool->get_window_shape(),
            avg_pool->get_window_movement_strides(),
            avg_pool->get_padding_below(), avg_pool->get_padding_above(),
            avg_pool->get_include_padding_in_avg_computation(), m_he_backend);
      } else {
        throw ngraph_error("Broadcast types not supported.");
      }
      break;
    }
    case OP_TYPEID::BatchNormInference: {
      const ngraph::op::BatchNormInference* bn =
          static_cast<const ngraph::op::BatchNormInference*>(&node);
      double eps = bn->get_eps_value();
      NGRAPH_ASSERT(args.size() == 5)
          << "BatchNormInference has " << args.size()
          << "arguments (expected 5).";

      auto shape = node.get_input_shape(2);
      // TODO: cleanup
      if (batch_size != 1) {
        shape[0] = shape[0] / batch_size;
      }

      auto gamma = dynamic_pointer_cast<HEPlainTensor>(args[0]);
      auto beta = dynamic_pointer_cast<HEPlainTensor>(args[1]);
      auto input = dynamic_pointer_cast<HECipherTensor>(args[2]);
      auto mean = dynamic_pointer_cast<HEPlainTensor>(args[3]);
      auto variance = dynamic_pointer_cast<HEPlainTensor>(args[4]);

      NGRAPH_ASSERT(out0_cipher != nullptr) << "BatchNorm output not cipher";
      NGRAPH_ASSERT(gamma != nullptr) << "BatchNorm gamma not plain";
      NGRAPH_ASSERT(beta != nullptr) << "BatchNorm beta not plain";
      NGRAPH_ASSERT(input != nullptr) << "BatchNorm input not cipher";
      NGRAPH_ASSERT(mean != nullptr) << "BatchNorm mean not plaintext";
      NGRAPH_ASSERT(variance != nullptr) << "BatchNorm variance not plaintext";

      runtime::he::kernel::batch_norm_inference(
          eps, gamma->get_elements(), beta->get_elements(),
          input->get_elements(), mean->get_elements(), variance->get_elements(),
          out0_cipher->get_elements(), shape, batch_size, m_he_backend);
      break;
    }
    case OP_TYPEID::Broadcast: {
      const op::Broadcast* broadcast = static_cast<const op::Broadcast*>(&node);
      AxisSet broadcast_axes = broadcast->get_broadcast_axes();

      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        Shape in_shape = arg0_cipher->get_shape();
        Shape out_shape = out0_cipher->get_shape();
        runtime::he::kernel::broadcast(arg0_cipher->get_elements(),
                                       out0_cipher->get_elements(), in_shape,
                                       out_shape, broadcast_axes);
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        Shape in_shape = arg0_plain->get_shape();
        Shape out_shape = out0_plain->get_shape();
        runtime::he::kernel::broadcast(arg0_plain->get_elements(),
                                       out0_plain->get_elements(), in_shape,
                                       out_shape, broadcast_axes);
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
        vector<Shape> in_shapes;
        vector<vector<shared_ptr<HECiphertext>>> in_args;

        for (shared_ptr<HETensor> arg : args) {
          shared_ptr<HECipherTensor> arg_cipher =
              dynamic_pointer_cast<HECipherTensor>(arg);
          if (arg_cipher == nullptr) {
            throw ngraph_error("Concat type not consistent");
          }
          in_args.push_back(arg_cipher->get_elements());
          in_shapes.push_back(arg_cipher->get_shape());

          runtime::he::kernel::concat(in_args, out0_cipher->get_elements(),
                                      in_shapes, node.get_output_shape(0),
                                      concat->get_concatenation_axis());
        }
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        vector<Shape> in_shapes;
        vector<vector<shared_ptr<HEPlaintext>>> in_args;

        for (shared_ptr<HETensor> arg : args) {
          shared_ptr<HEPlainTensor> arg_plain =
              dynamic_pointer_cast<HEPlainTensor>(arg);
          if (arg_plain == nullptr) {
            throw ngraph_error("Concat type not consistent");
          }
          in_args.push_back(arg_plain->get_elements());
          in_shapes.push_back(arg_plain->get_shape());

          runtime::he::kernel::concat(in_args, out0_plain->get_elements(),
                                      in_shapes, node.get_output_shape(0),
                                      concat->get_concatenation_axis());
        }
      } else {
        throw ngraph_error("Concat types not supported.");
      }
      break;
    }
    case OP_TYPEID::Constant: {
      const op::Constant* constant = static_cast<const op::Constant*>(&node);

      if (out0_plain != nullptr) {
        runtime::he::kernel::constant(out0_plain->get_elements(), type,
                                      constant->get_data_ptr(), m_he_backend,
                                      out0_plain->get_batched_element_count());
      } else if (out0_cipher != nullptr) {
        runtime::he::kernel::constant(out0_cipher->get_elements(), type,
                                      constant->get_data_ptr(), m_he_backend,
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

      if (arg0_cipher != nullptr && arg1_cipher != nullptr &&
          out0_cipher != nullptr) {
        runtime::he::kernel::convolution(
            arg0_cipher->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), arg0_cipher->get_batched_shape(),
            arg1_cipher->get_batched_shape(), out0_cipher->get_batched_shape(),
            window_movement_strides, window_dilation_strides, padding_below,
            padding_above, data_dilation_strides, 0, 1, 1, 0, 0, 1, false, type,
            batch_size, m_he_backend);
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::convolution(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), arg0_cipher->get_batched_shape(),
            arg1_plain->get_batched_shape(), out0_cipher->get_batched_shape(),
            window_movement_strides, window_dilation_strides, padding_below,
            padding_above, data_dilation_strides, 0, 1, 1, 0, 0, 1, false, type,
            batch_size, m_he_backend);
      } else if (arg0_plain != nullptr && arg1_cipher != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::convolution(
            arg0_plain->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), arg0_plain->get_batched_shape(),
            arg1_cipher->get_batched_shape(), out0_cipher->get_batched_shape(),
            window_movement_strides, window_dilation_strides, padding_below,
            padding_above, data_dilation_strides, 0, 1, 1, 0, 0, 1, false, type,
            batch_size, m_he_backend);
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        runtime::he::kernel::convolution(
            arg0_plain->get_elements(), arg1_plain->get_elements(),
            out0_plain->get_elements(), arg0_plain->get_batched_shape(),
            arg1_plain->get_batched_shape(), out0_plain->get_batched_shape(),
            window_movement_strides, window_dilation_strides, padding_below,
            padding_above, data_dilation_strides, 0, 1, 1, 0, 0, 1, false, type,
            batch_size, m_he_backend);
      } else {
        throw ngraph_error("Convolution types not supported.");
      }
      break;
    }
    case OP_TYPEID::Dot: {
      const op::Dot* dot = static_cast<const op::Dot*>(&node);

      NGRAPH_INFO << join(args[0]->get_batched_shape(), "x") << " dot "
                  << join(args[1]->get_batched_shape(), "x");
      if (arg0_cipher != nullptr && arg1_cipher != nullptr &&
          out0_cipher != nullptr) {
        runtime::he::kernel::dot(
            arg0_cipher->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), arg0_cipher->get_batched_shape(),
            arg1_cipher->get_batched_shape(), out0_cipher->get_batched_shape(),
            dot->get_reduction_axes_count(), type, m_he_backend);
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::dot(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), arg0_cipher->get_batched_shape(),
            arg1_plain->get_batched_shape(), out0_cipher->get_batched_shape(),
            dot->get_reduction_axes_count(), type, m_he_backend);
      } else if (arg0_plain != nullptr && arg1_cipher != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::dot(
            arg0_plain->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), arg0_plain->get_batched_shape(),
            arg1_cipher->get_batched_shape(), out0_cipher->get_batched_shape(),
            dot->get_reduction_axes_count(), type, m_he_backend);
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        runtime::he::kernel::dot(
            arg0_plain->get_elements(), arg1_plain->get_elements(),
            out0_plain->get_elements(), arg0_plain->get_batched_shape(),
            arg1_plain->get_batched_shape(), out0_plain->get_batched_shape(),
            dot->get_reduction_axes_count(), type, m_he_backend);
      } else {
        throw ngraph_error("Dot types not supported.");
      }
      break;
    }
    case OP_TYPEID::MaxPool: {
      if (!m_enable_client) {
        throw ngraph_error(
            "MaxPool op unsupported unless client is enabled. Try setting "
            "NGRAPH_ENABLE_CLIENT=1");
      }
      if (arg0_cipher == nullptr || out0_cipher == nullptr) {
        NGRAPH_INFO << "MaxPool types not supported ";
        throw ngraph_error("MaxPool supports only Cipher, Cipher");
      }
      m_max_ciphertexts.clear();

      NGRAPH_INFO << "MaxPool shape " << join(node.get_output_shape(0), "x");
      NGRAPH_INFO << "m_batch_size " << m_batch_size;

      const op::MaxPool* max_pool = static_cast<const op::MaxPool*>(&node);
      // TODO: cleanup
      Shape arg0_shape = node.get_inputs().at(0).get_shape();
      arg0_shape[0] /= m_batch_size;
      Shape out_shape = node.get_output_shape(0);
      out_shape[0] /= m_batch_size;

      vector<vector<size_t>> maximize_list = runtime::he::kernel::max_pool(
          arg0_shape, out_shape, max_pool->get_window_shape(),
          max_pool->get_window_movement_strides(),
          max_pool->get_padding_below(), max_pool->get_padding_above());

      for (size_t list_ind = 0; list_ind < maximize_list.size(); list_ind++) {
        stringstream cipher_stream;
        size_t cipher_count = 0;

        for (const size_t max_ind : maximize_list[list_ind]) {
          auto he_ciphertext = arg0_cipher->get_element(max_ind);
          he_ciphertext->save(cipher_stream);
          cipher_count++;
        }
        // Send list of ciphertexts to maximize over to client
        /*NGRAPH_INFO << "Sending " << cipher_count
                    << " Maxpool ciphertexts (size "
                    << cipher_stream.str().size() << ") to client"; */
        auto max_message =
            TCPMessage(MessageType::max_request, cipher_count, cipher_stream);

        m_session->do_write(max_message);

        // Acquire lock
        unique_lock<mutex> mlock(m_max_mutex);

        // Wait until max is done
        m_max_cond.wait(mlock, std::bind(&HEExecutable::max_done, this));

        // Reset for next max call
        m_max_done = false;
      }

      out0_cipher->set_elements(m_max_ciphertexts);
      break;
    }
    case OP_TYPEID::Minimum: {
      NGRAPH_INFO << "Minimum op";
      if (!m_enable_client) {
        throw ngraph_error(
            "Minimum op unsupported unless client is enabled. Try setting "
            "NGRAPH_ENABLE_CLIENT=1");
      }
      if (out0_cipher == nullptr) {
        NGRAPH_INFO << "Minimum types not supported ";
        throw ngraph_error("Minimum supports only output cipher");
      }

      size_t element_count =
          shape_size(node.get_output_shape(0)) / m_batch_size;

      if (arg0_plain != nullptr) {
        // arg0_plain doesn't have a tensor layout, so we use
        const element::Type& element_type =
            out0_cipher->get_tensor_layout()->get_element_type();
        arg0_cipher = dynamic_pointer_cast<HECipherTensor>(
            m_he_backend->create_cipher_tensor(
                element_type, arg0_plain->get_shape(), m_batch_data));
        for (size_t elem_idx = 0; elem_idx < element_count; ++elem_idx) {
          m_he_backend->encrypt(arg0_cipher->get_element(elem_idx),
                                (arg0_plain->get_element(elem_idx).get()));
        }
      }
      if (arg1_plain != nullptr) {
        // arg0_plain doesn't have a tensor layout, so we use
        const element::Type& element_type =
            out0_cipher->get_tensor_layout()->get_element_type();
        arg1_cipher = dynamic_pointer_cast<HECipherTensor>(
            m_he_backend->create_cipher_tensor(
                element_type, arg1_plain->get_shape(), m_batch_data));
        for (size_t elem_idx = 0; elem_idx < element_count; ++elem_idx) {
          m_he_backend->encrypt(arg1_cipher->get_element(elem_idx),
                                (arg1_plain->get_element(elem_idx).get()));
        }
      }
      NGRAPH_ASSERT(arg0_cipher != nullptr) << "arg0_cipher is nullptr";
      NGRAPH_ASSERT(arg1_cipher != nullptr) << "arg1_cipher is nullptr";

      m_minimum_ciphertexts.clear();

      NGRAPH_INFO << "Min shape " << join(node.get_output_shape(0), "x");

      // TODO: cleanup
      Shape arg0_shape = node.get_inputs().at(0).get_shape();
      arg0_shape[0] /= m_batch_size;
      Shape out_shape = node.get_output_shape(0);
      out_shape[0] /= m_batch_size;

      NGRAPH_ASSERT(arg0_cipher->get_elements().size() ==
                    arg1_cipher->get_elements().size())
          << "Element counts " << arg0_cipher->get_elements().size() << ",  "
          << arg1_cipher->get_elements().size() << "do not match";

      stringstream cipher_stream;
      size_t cipher_count = 0;
      auto he_ckks_backend =
          (runtime::he::he_seal::HESealCKKSBackend*)m_he_backend;
      NGRAPH_ASSERT(he_ckks_backend != nullptr)
          << "HEBackend is not CKKS in Minimum Op";
      for (size_t min_ind = 0; min_ind < element_count; ++min_ind) {
        auto cipher0 = arg0_cipher->get_element(min_ind);
        auto cipher1 = arg1_cipher->get_element(min_ind);
        cipher0->save(cipher_stream);
        cipher1->save(cipher_stream);
        cipher_count += 2;
      }

      // Send list of ciphertexts to minimum over to client
      NGRAPH_INFO << "Sending " << cipher_count << " Minimum ciphertexts (size "
                  << cipher_stream.str().size() << ") to client";
      auto minimum_message =
          TCPMessage(MessageType::minimum_request, cipher_count, cipher_stream);

      m_session->do_write(minimum_message);

      // Acquire lock
      unique_lock<mutex> mlock(m_minimum_mutex);

      // Wait until minimum is done
      m_minimum_cond.wait(mlock, std::bind(&HEExecutable::minimum_done, this));

      // Reset for next max call
      m_minimum_done = false;

      out0_cipher->set_elements(m_minimum_ciphertexts);
      break;
    }
    case OP_TYPEID::Multiply: {
      if (arg0_cipher != nullptr && arg1_cipher != nullptr &&
          out0_cipher != nullptr) {
        runtime::he::kernel::multiply(
            arg0_cipher->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), type, m_he_backend,
            out0_cipher->get_batched_element_count());
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::multiply(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), type, m_he_backend,
            out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && arg1_cipher != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::multiply(
            arg0_plain->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), type, m_he_backend,
            out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        runtime::he::kernel::multiply(
            arg0_plain->get_elements(), arg1_plain->get_elements(),
            out0_plain->get_elements(), type, m_he_backend,
            out0_plain->get_batched_element_count());
      } else {
        throw ngraph_error("Multiply types not supported.");
      }
      break;
    }
    case OP_TYPEID::Negative: {
      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        runtime::he::kernel::negate(
            arg0_cipher->get_elements(), out0_cipher->get_elements(), type,
            m_he_backend, out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        runtime::he::kernel::negate(
            arg0_plain->get_elements(), out0_plain->get_elements(), type,
            m_he_backend, out0_plain->get_batched_element_count());
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
      if (arg0_cipher != nullptr && arg1_cipher != nullptr &&
          out0_cipher != nullptr) {
        runtime::he::kernel::pad(
            arg0_cipher->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), arg0_cipher->get_batched_shape(),
            out0_cipher->get_batched_shape(), pad->get_padding_below(),
            pad->get_padding_above(), pad->get_pad_mode(), batch_size,
            m_he_backend);
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::pad(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), arg0_cipher->get_batched_shape(),
            out0_cipher->get_batched_shape(), pad->get_padding_below(),
            pad->get_padding_above(), pad->get_pad_mode(), batch_size,
            m_he_backend);
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        runtime::he::kernel::pad(
            arg0_plain->get_elements(), arg1_plain->get_elements(),
            out0_plain->get_elements(), arg0_plain->get_batched_shape(),
            out0_plain->get_batched_shape(), pad->get_padding_below(),
            pad->get_padding_above(), pad->get_pad_mode(), batch_size,
            m_he_backend);
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
    case OP_TYPEID::Reshape: {
      NGRAPH_INFO << "Reshape op";
      const op::Reshape* reshape = static_cast<const op::Reshape*>(&node);
      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        runtime::he::kernel::reshape(
            arg0_cipher->get_elements(), out0_cipher->get_elements(),
            arg0_cipher->get_batched_shape(), reshape->get_input_order(),
            out0_cipher->get_batched_shape());
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        runtime::he::kernel::reshape(
            arg0_plain->get_elements(), out0_plain->get_elements(),
            arg0_plain->get_batched_shape(), reshape->get_input_order(),
            out0_plain->get_batched_shape());
      } else {
        throw ngraph_error("Reshape types not supported.");
      }
      NGRAPH_INFO << "Done with reshape op";
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
        runtime::he::kernel::result(arg0_cipher->get_elements(),
                                    out0_cipher->get_elements(), output_size);
      } else if (arg0_plain != nullptr && out0_cipher != nullptr) {
        runtime::he::kernel::result(arg0_plain->get_elements(),
                                    out0_cipher->get_elements(), output_size,
                                    m_he_backend);
      } else if (arg0_cipher != nullptr && out0_plain != nullptr) {
        runtime::he::kernel::result(arg0_cipher->get_elements(),
                                    out0_plain->get_elements(), output_size,
                                    m_he_backend);
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        runtime::he::kernel::result(arg0_plain->get_elements(),
                                    out0_plain->get_elements(), output_size);
      } else {
        throw ngraph_error("Result types not supported.");
      }
      break;
    }
    case OP_TYPEID::Relu: {
      if (!m_enable_client) {
        throw ngraph_error(
            "Relu op unsupported unless client is enabled. Try setting "
            "NGRAPH_ENABLE_CLIENT=1");
      }

      size_t element_count =
          shape_size(node.get_output_shape(0)) / m_batch_size;

      if (arg0_cipher == nullptr || out0_cipher == nullptr) {
        NGRAPH_INFO << "Relu types not supported ";
        throw ngraph_error("Relu types not supported.");
      }

      stringstream cipher_stream;
      arg0_cipher->save_elements(cipher_stream);

      // Send output to client
      NGRAPH_INFO << "Sending " << element_count << " Relu ciphertexts (size "
                  << cipher_stream.str().size() << ") to client";
      auto relu_message =
          TCPMessage(MessageType::relu_request, element_count, cipher_stream);

      m_session->do_write(relu_message);

      // Acquire lock
      unique_lock<mutex> mlock(m_relu_mutex);

      // Wait until Relu is done
      m_relu_cond.wait(mlock, std::bind(&HEExecutable::relu_done, this));
      NGRAPH_INFO << "Relu is done";

      // Reset for next Relu call
      m_relu_done = false;
      out0_cipher->set_elements(m_relu_ciphertexts);
      break;
    }
    case OP_TYPEID::Reverse: {
      const op::Reverse* reverse = static_cast<const op::Reverse*>(&node);
      Shape in_shape = node.get_input_shape(0);
      Shape out_shape = node.get_output_shape(0);

      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        runtime::he::kernel::reverse(arg0_cipher->get_elements(),
                                     out0_cipher->get_elements(), in_shape,
                                     out_shape, reverse->get_reversed_axes());
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        runtime::he::kernel::reverse(arg0_plain->get_elements(),
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
      Shape in_shape = node.get_input_shape(0);
      Shape out_shape = node.get_output_shape(0);

      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        runtime::he::kernel::slice(
            arg0_cipher->get_elements(), out0_cipher->get_elements(), in_shape,
            slice->get_lower_bounds(), slice->get_upper_bounds(),
            slice->get_strides(), out_shape);
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        runtime::he::kernel::slice(
            arg0_plain->get_elements(), out0_plain->get_elements(), in_shape,
            slice->get_lower_bounds(), slice->get_upper_bounds(),
            slice->get_strides(), out_shape);
      } else {
        throw ngraph_error("Slice types not supported.");
      }
      break;
    }
    case OP_TYPEID::Subtract: {
      if (arg0_cipher != nullptr && arg1_cipher != nullptr &&
          out0_cipher != nullptr) {
        runtime::he::kernel::subtract(
            arg0_cipher->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), type, m_he_backend,
            out0_cipher->get_batched_element_count());
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::subtract(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), type, m_he_backend,
            out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && arg1_cipher != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::subtract(
            arg0_plain->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), type, m_he_backend,
            out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        runtime::he::kernel::subtract(
            arg0_plain->get_elements(), arg1_plain->get_elements(),
            out0_plain->get_elements(), type, m_he_backend,
            out0_plain->get_batched_element_count());
      } else {
        throw ngraph_error("Subtract types not supported.");
      }
      break;
    }
    case OP_TYPEID::Sum: {
      const op::Sum* sum = static_cast<const op::Sum*>(&node);
      Shape in_shape = node.get_input_shape(0);
      Shape out_shape = node.get_output_shape(0);

      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        runtime::he::kernel::sum(
            arg0_cipher->get_elements(), out0_cipher->get_elements(), in_shape,
            out_shape, sum->get_reduction_axes(), type, m_he_backend);
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        runtime::he::kernel::sum(
            arg0_plain->get_elements(), out0_plain->get_elements(), in_shape,
            out_shape, sum->get_reduction_axes(), type, m_he_backend);
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
    case OP_TYPEID::EmbeddingLookup:
    case OP_TYPEID::Equal:
    case OP_TYPEID::Erf:
    case OP_TYPEID::Exp:
    case OP_TYPEID::Floor:
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
    case OP_TYPEID::ReluBackprop:
    case OP_TYPEID::ReplaceSlice:
    case OP_TYPEID::ReverseSequence:
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
    case OP_TYPEID::TopK:
    case OP_TYPEID::Transpose:
    default:
      throw unsupported_op("Unsupported op '" + node.description() + "'");
#pragma GCC diagnostic pop
  }
}
