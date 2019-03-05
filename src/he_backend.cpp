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

#include "he_backend.hpp"
#include "he_cipher_tensor.hpp"
#include "he_plain_tensor.hpp"
#include "he_tensor.hpp"
#include "kernel/add.hpp"
#include "kernel/avg_pool.hpp"
#include "kernel/broadcast.hpp"
#include "kernel/concat.hpp"
#include "kernel/constant.hpp"
#include "kernel/convolution.hpp"
#include "kernel/dot.hpp"
#include "kernel/multiply.hpp"
#include "kernel/negate.hpp"
#include "kernel/pad.hpp"
#include "kernel/reshape.hpp"
#include "kernel/result.hpp"
#include "kernel/reverse.hpp"
#include "kernel/slice.hpp"
#include "kernel/subtract.hpp"
#include "kernel/sum.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/function.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"

using namespace ngraph;
using namespace std;

using descriptor::layout::DenseTensorLayout;

void runtime::he::HEBackend::start_server() {
  // Server
  tcp::resolver resolver(m_io_context);
  tcp::endpoint server_endpoints(tcp::v4(), m_port);
  auto server_callback = [this](const runtime::he::TCPMessage message) {
    handle_message(message);
  };
  m_tcp_server =
      make_shared<TCPServer>(m_io_context, server_endpoints, server_callback);
  // m_thread = std::thread([&io_context]() { io_context.run(); });
  m_io_context.run();  // Actually start the server
}

shared_ptr<runtime::he::HEPlaintext>
runtime::he::HEBackend::create_valued_plaintext(
    float value, const element::Type& element_type) const {
  const string type_name = element_type.c_type_string();
  shared_ptr<runtime::he::HEPlaintext> plaintext = create_empty_plaintext();

  encode(plaintext, (void*)(&value), element_type, 1);
  return plaintext;
}

shared_ptr<runtime::he::HECiphertext>
runtime::he::HEBackend::create_valued_ciphertext(
    float value, const element::Type& element_type, size_t batch_size) const {
  if (batch_size != 1) {
    throw ngraph_error(
        "HEBackend::create_valued_ciphertext only supports batch size 1");
  }
  const string type_name = element_type.c_type_string();
  shared_ptr<runtime::he::HEPlaintext> plaintext =
      create_valued_plaintext(value, element_type);
  shared_ptr<runtime::he::HECiphertext> ciphertext = create_empty_ciphertext();

  encrypt(ciphertext, *plaintext);
  return ciphertext;
}

shared_ptr<runtime::Tensor> runtime::he::HEBackend::create_tensor(
    const element::Type& element_type, const Shape& shape,
    void* memory_pointer) {
  throw ngraph_error("HE create_tensor unimplemented");
}

shared_ptr<runtime::Tensor> runtime::he::HEBackend::create_tensor(
    const element::Type& element_type, const Shape& shape) {
  if (batch_data()) {
    return create_batched_plain_tensor(element_type, shape);
  } else {
    return create_plain_tensor(element_type, shape);
  }
}

shared_ptr<runtime::Tensor> runtime::he::HEBackend::create_plain_tensor(
    const element::Type& element_type, const Shape& shape,
    const bool batched) const {
  auto rc = make_shared<runtime::he::HEPlainTensor>(
      element_type, shape, this, create_empty_plaintext(), batched);
  return static_pointer_cast<runtime::Tensor>(rc);
}

shared_ptr<runtime::Tensor> runtime::he::HEBackend::create_cipher_tensor(
    const element::Type& element_type, const Shape& shape,
    const bool batched) const {
  auto rc = make_shared<runtime::he::HECipherTensor>(
      element_type, shape, this, create_empty_ciphertext(), batched);
  return static_pointer_cast<runtime::Tensor>(rc);
}

shared_ptr<runtime::Tensor> runtime::he::HEBackend::create_valued_cipher_tensor(
    float value, const element::Type& element_type, const Shape& shape) const {
  auto tensor = static_pointer_cast<HECipherTensor>(
      create_cipher_tensor(element_type, shape));
  vector<shared_ptr<runtime::he::HECiphertext>>& cipher_texts =
      tensor->get_elements();
#pragma omp parallel for
  for (size_t i = 0; i < cipher_texts.size(); ++i) {
    cipher_texts[i] = create_valued_ciphertext(value, element_type);
  }
  return tensor;
}

shared_ptr<runtime::Tensor> runtime::he::HEBackend::create_valued_plain_tensor(
    float value, const element::Type& element_type, const Shape& shape) const {
  auto tensor = static_pointer_cast<HEPlainTensor>(
      create_plain_tensor(element_type, shape));
  vector<shared_ptr<runtime::he::HEPlaintext>>& plain_texts =
      tensor->get_elements();
#pragma omp parallel for
  for (size_t i = 0; i < plain_texts.size(); ++i) {
    plain_texts[i] = create_valued_plaintext(value, element_type);
  }
  return tensor;
}

runtime::Handle runtime::he::HEBackend::compile(shared_ptr<Function> function) {
  NGRAPH_INFO << "Compiling function";
  FunctionInstance& instance = m_function_map[function];
  if (!instance.m_is_compiled) {
    instance.m_is_compiled = true;
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::LikeReplacement>();
    pass_manager.register_pass<pass::AssignLayout<DenseTensorLayout>>();
    pass_manager.register_pass<pass::Liveness>();
    pass_manager.run_passes(function);

    for (const shared_ptr<Node>& node : function->get_ordered_ops()) {
      instance.m_wrapped_nodes.emplace_back(node);
    }
  }
  NGRAPH_INFO << "Compiled function";
  return function;
}

void runtime::he::HEBackend::validate_he_call(
    shared_ptr<const Function> function,
    const vector<shared_ptr<runtime::he::HETensor>>& outputs,
    const vector<shared_ptr<runtime::he::HETensor>>& inputs) {
  const ParameterVector& input_parameters = function->get_parameters();
  if (input_parameters.size() != inputs.size()) {
    stringstream ss;
    ss << "Call input count " << inputs.size()
       << " does not match Function's Parameter count "
       << input_parameters.size();
    throw runtime_error(ss.str());
  }
  if (function->get_output_size() != outputs.size()) {
    stringstream ss;
    ss << "Call output count " << outputs.size()
       << " does not match Function's Result count "
       << function->get_output_size();
    throw runtime_error(ss.str());
  }

  for (size_t i = 0; i < input_parameters.size(); i++) {
    if (inputs[i]->get_element_type() !=
        input_parameters[i]->get_element_type()) {
      stringstream ss;
      ss << "Input " << i << " type '" << inputs[i]->get_element_type()
         << "' does not match Parameter type '"
         << input_parameters[i]->get_element_type() << "'";
      throw runtime_error(ss.str());
    }
    if (inputs[i]->get_expanded_shape() != input_parameters[i]->get_shape()) {
      stringstream ss;
      ss << "Input " << i << " shape {" << join(inputs[i]->get_expanded_shape())
         << "} does not match Parameter shape {"
         << join(input_parameters[i]->get_shape()) << "}";
      throw runtime_error(ss.str());
    }
  }

  for (size_t i = 0; i < function->get_output_size(); i++) {
    if (outputs[i]->get_element_type() !=
        function->get_output_element_type(i)) {
      stringstream ss;
      ss << "Output " << i << " type '" << outputs[i]->get_element_type()
         << "' does not match Result type '"
         << function->get_output_element_type(i) << "'";
      throw runtime_error(ss.str());
    }
    if (function->get_output_shape(i) != outputs[i]->get_expanded_shape()) {
      stringstream ss;
      ss << "Output " << i << " shape {"
         << join(outputs[i]->get_expanded_shape())
         << "} does not match Result shape {"
         << join(function->get_output_shape(i)) << "}";
      throw runtime_error(ss.str());
    }
  }
}

void runtime::he::HEBackend::clear_function_instance() {
  m_function_map.clear();
}

void runtime::he::HEBackend::remove_compiled_function(
    shared_ptr<Function> function) {
  m_function_map.erase(function);
}

void runtime::he::HEBackend::enable_performance_data(
    shared_ptr<Function> function, bool enable) {
  // Enabled by default
}

vector<runtime::PerformanceCounter>
runtime::he::HEBackend::get_performance_data(
    shared_ptr<Function> function) const {
  vector<runtime::PerformanceCounter> rc;
  const FunctionInstance& instance = m_function_map.at(function);
  for (const pair<const Node*, stopwatch> p : instance.m_timer_map) {
    rc.emplace_back(p.first->get_name().c_str(),
                    p.second.get_total_microseconds(),
                    p.second.get_call_count());
  }
  return rc;
}

bool runtime::he::HEBackend::call(
    shared_ptr<Function> function,
    const vector<shared_ptr<runtime::Tensor>>& outputs,
    const vector<shared_ptr<runtime::Tensor>>& inputs) {
  if (encrypt_data()) {
    NGRAPH_INFO << "Encrypting data";
  }
  if (batch_data()) {
    NGRAPH_INFO << "Batching data";
  }
  if (encrypt_model()) {
    NGRAPH_INFO << "Encrypting model";
  }

  auto fit = m_function_map.find(function);
  if (fit == m_function_map.end()) {
    throw runtime_error("compile() must be called before call().");
  }
  FunctionInstance& instance = fit->second;

  // convert outputs to HETensor
  vector<shared_ptr<runtime::he::HETensor>> he_inputs;
  for (auto& tv : inputs) {
    he_inputs.push_back(static_pointer_cast<runtime::he::HETensor>(tv));
  }

  // convert inputs to HETensor
  vector<shared_ptr<runtime::he::HETensor>> he_outputs;
  for (auto& tv : outputs) {
    he_outputs.push_back(static_pointer_cast<runtime::he::HETensor>(tv));
  }

  validate_he_call(function, he_outputs, he_inputs);

  // map function params -> HETensor
  unordered_map<descriptor::Tensor*, shared_ptr<runtime::he::HETensor>>
      tensor_map;
  size_t input_count = 0;
  for (auto param : function->get_parameters()) {
    for (size_t i = 0; i < param->get_output_size(); ++i) {
      descriptor::Tensor* tv = param->get_output_tensor_ptr(i).get();

      if (encrypt_data()) {
        NGRAPH_INFO << "Encrypting parameter " << i;
        auto plain_input = static_pointer_cast<runtime::he::HEPlainTensor>(
            he_inputs[input_count]);
        assert(plain_input != nullptr);
        auto cipher_input = static_pointer_cast<runtime::he::HECipherTensor>(
            create_cipher_tensor(plain_input->get_element_type(),
                                 plain_input->get_shape(), batch_data()));

        NGRAPH_INFO << "plain_input->get_batched_element_count() "
                    << plain_input->get_batched_element_count();
#pragma omp parallel for
        for (size_t i = 0; i < plain_input->get_batched_element_count(); ++i) {
          encrypt(cipher_input->get_element(i), *plain_input->get_element(i));
        }

        NGRAPH_INFO << "Done encrypting parameter " << i;

        tensor_map.insert({tv, cipher_input});
        input_count++;
      } else {
        tensor_map.insert({tv, he_inputs[input_count++]});
      }
    }
  }

  // map function outputs -> HostTensor
  for (size_t output_count = 0; output_count < function->get_output_size();
       ++output_count) {
    auto output = function->get_output_op(output_count);
    if (!dynamic_pointer_cast<op::Result>(output)) {
      throw ngraph_error("One of function's outputs isn't op::Result");
    }
    descriptor::Tensor* tv = output->get_output_tensor_ptr(0).get();
    tensor_map.insert({tv, he_outputs[output_count++]});
  }

  // for each ordered op in the graph
  for (const NodeWrapper& wrapped : instance.m_wrapped_nodes) {
    const Node* op = &wrapped.get_node();
    auto type_id = wrapped.get_typeid();

    NGRAPH_INFO << "\033[1;32m"
                << "[ " << op->get_name() << " ]"
                << "\033[0m";

    if (type_id == OP_TYPEID::Parameter) {
      NGRAPH_INFO << "Parameter shape {" << join(op->get_shape()) << "}";
      continue;
    }

    if (op->description() == "Constant") {
      NGRAPH_INFO << "Constant shape {" << join(op->get_shape()) << "}";
    }

    // get op inputs from map
    vector<shared_ptr<runtime::he::HETensor>> op_inputs;
    for (const descriptor::Input& input : op->get_inputs()) {
      descriptor::Tensor* tv = input.get_output().get_tensor_ptr().get();
      op_inputs.push_back(tensor_map.at(tv));
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

        bool plain_out = all_of(
            op_inputs.begin(), op_inputs.end(),
            [](shared_ptr<runtime::he::HETensor> op_input) {
              return dynamic_pointer_cast<HEPlainTensor>(op_input) != nullptr;
            });
        if (op->is_constant()) {
          plain_out = !encrypt_model();
        }

        bool batched_out = any_of(op_inputs.begin(), op_inputs.end(),
                                  [](shared_ptr<runtime::he::HETensor> he_tv) {
                                    return he_tv->is_batched();
                                  });
        if (plain_out) {
          auto otv = make_shared<runtime::he::HEPlainTensor>(
              element_type, shape, this, create_empty_plaintext(), batched_out,
              name);
          tensor_map.insert({tv, otv});
        } else {
          auto otv = make_shared<runtime::he::HECipherTensor>(
              element_type, shape, this, create_empty_ciphertext(), batched_out,
              name);
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

    instance.m_timer_map[op].start();
    generate_calls(base_type, wrapped, op_outputs, op_inputs, instance);
    instance.m_timer_map[op].stop();

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
                << instance.m_timer_map[op].get_milliseconds() << "ms"
                << "\033[0m";
  }
  size_t total_time = 0;
  for (const auto& elem : instance.m_timer_map) {
    total_time += elem.second.get_milliseconds();
  }
  NGRAPH_INFO << "\033[1;32m"
              << "Total time " << total_time << " (ms) \033[0m";
  return true;
}

void runtime::he::HEBackend::generate_calls(
    const element::Type& element_type, const NodeWrapper& node_wrapper,
    const vector<shared_ptr<HETensor>>& out,
    const vector<shared_ptr<HETensor>>& args, FunctionInstance& instance) {
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
  }
  if (args.size() > 1) {
    arg1_cipher = dynamic_pointer_cast<HECipherTensor>(args[1]);
    arg1_plain = dynamic_pointer_cast<HEPlainTensor>(args[1]);
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
            out0_cipher->get_elements(), element_type, this,
            out0_cipher->get_batched_element_count());
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        NGRAPH_INFO << "Add cipher+plain";
        runtime::he::kernel::add(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), element_type, this,
            out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && arg1_cipher != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::add(
            arg0_plain->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), element_type, this,
            out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        runtime::he::kernel::add(arg0_plain->get_elements(),
                                 arg1_plain->get_elements(),
                                 out0_plain->get_elements(), element_type, this,
                                 out0_plain->get_batched_element_count());
      } else {
        throw ngraph_error("Add types not supported.");
      }
      break;
    }
    case OP_TYPEID::AvgPool: {
      const op::AvgPool* avg_pool = static_cast<const op::AvgPool*>(&node);
      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        Shape in_shape = arg0_cipher->get_shape();
        Shape out_shape = out0_cipher->get_shape();
        runtime::he::kernel::avg_pool(
            arg0_cipher->get_elements(), out0_cipher->get_elements(), in_shape,
            out_shape, avg_pool->get_window_shape(),
            avg_pool->get_window_movement_strides(),
            avg_pool->get_padding_below(), avg_pool->get_padding_above(),
            avg_pool->get_include_padding_in_avg_computation(), this);

      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        Shape in_shape = arg0_plain->get_shape();
        Shape out_shape = out0_plain->get_shape();
        runtime::he::kernel::avg_pool(
            arg0_plain->get_elements(), out0_plain->get_elements(), in_shape,
            out_shape, avg_pool->get_window_shape(),
            avg_pool->get_window_movement_strides(),
            avg_pool->get_padding_below(), avg_pool->get_padding_above(),
            avg_pool->get_include_padding_in_avg_computation(), this);
      } else {
        throw ngraph_error("Broadcast types not supported.");
      }
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
        std::vector<Shape> in_shapes;
        std::vector<std::vector<std::shared_ptr<HECiphertext>>> in_args;

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
        std::vector<Shape> in_shapes;
        std::vector<std::vector<std::shared_ptr<HEPlaintext>>> in_args;

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
        runtime::he::kernel::constant(out0_plain->get_elements(), element_type,
                                      constant->get_data_ptr(), this,
                                      out0_plain->get_batched_element_count());
      } else if (out0_cipher != nullptr) {
        runtime::he::kernel::constant(out0_cipher->get_elements(), element_type,
                                      constant->get_data_ptr(), this,
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
            out0_cipher->get_elements(), arg0_cipher->get_shape(),
            arg1_cipher->get_shape(), out0_cipher->get_shape(),
            window_movement_strides, window_dilation_strides, padding_below,
            padding_above, data_dilation_strides, 0, 1, 1, 0, 0, 1, false,
            element_type, batch_size, this);
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::convolution(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), arg0_cipher->get_shape(),
            arg1_plain->get_shape(), out0_cipher->get_shape(),
            window_movement_strides, window_dilation_strides, padding_below,
            padding_above, data_dilation_strides, 0, 1, 1, 0, 0, 1, false,
            element_type, batch_size, this);
      } else if (arg0_plain != nullptr && arg1_cipher != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::convolution(
            arg0_plain->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), arg0_plain->get_shape(),
            arg1_cipher->get_shape(), out0_cipher->get_shape(),
            window_movement_strides, window_dilation_strides, padding_below,
            padding_above, data_dilation_strides, 0, 1, 1, 0, 0, 1, false,
            element_type, batch_size, this);
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        runtime::he::kernel::convolution(
            arg0_plain->get_elements(), arg1_plain->get_elements(),
            out0_plain->get_elements(), arg0_plain->get_shape(),
            arg1_plain->get_shape(), out0_plain->get_shape(),
            window_movement_strides, window_dilation_strides, padding_below,
            padding_above, data_dilation_strides, 0, 1, 1, 0, 0, 1, false,
            element_type, batch_size, this);
      } else {
        throw ngraph_error("Convolution types not supported.");
      }
      break;
    }
    case OP_TYPEID::Dot: {
      const op::Dot* dot = static_cast<const op::Dot*>(&node);

      NGRAPH_INFO << join(args[0]->get_shape(), "x") << " dot "
                  << join(args[1]->get_shape(), "x");
      if (arg0_cipher != nullptr && arg1_cipher != nullptr &&
          out0_cipher != nullptr) {
        runtime::he::kernel::dot(
            arg0_cipher->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), arg0_cipher->get_shape(),
            arg1_cipher->get_shape(), out0_cipher->get_shape(),
            dot->get_reduction_axes_count(), element_type, this);
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::dot(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), arg0_cipher->get_shape(),
            arg1_plain->get_shape(), out0_cipher->get_shape(),
            dot->get_reduction_axes_count(), element_type, this);
      } else if (arg0_plain != nullptr && arg1_cipher != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::dot(
            arg0_plain->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), arg0_plain->get_shape(),
            arg1_cipher->get_shape(), out0_cipher->get_shape(),
            dot->get_reduction_axes_count(), element_type, this);
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        runtime::he::kernel::dot(
            arg0_plain->get_elements(), arg1_plain->get_elements(),
            out0_plain->get_elements(), arg0_plain->get_shape(),
            arg1_plain->get_shape(), out0_plain->get_shape(),
            dot->get_reduction_axes_count(), element_type, this);
      } else {
        throw ngraph_error("Dot types not supported.");
      }
      break;
    }
    case OP_TYPEID::Multiply: {
      if (arg0_cipher != nullptr && arg1_cipher != nullptr &&
          out0_cipher != nullptr) {
        runtime::he::kernel::multiply(
            arg0_cipher->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), element_type, this,
            out0_cipher->get_batched_element_count());
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::multiply(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), element_type, this,
            out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && arg1_cipher != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::multiply(
            arg0_plain->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), element_type, this,
            out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        runtime::he::kernel::multiply(
            arg0_plain->get_elements(), arg1_plain->get_elements(),
            out0_plain->get_elements(), element_type, this,
            out0_plain->get_batched_element_count());
      } else {
        throw ngraph_error("Multiply types not supported.");
      }
      break;
    }
    case OP_TYPEID::Negative: {
      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        runtime::he::kernel::negate(
            arg0_cipher->get_elements(), out0_cipher->get_elements(),
            element_type, this, out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        runtime::he::kernel::negate(
            arg0_plain->get_elements(), out0_plain->get_elements(),
            element_type, this, out0_plain->get_batched_element_count());
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

      // TODO: clean up
      Shape arg0_shape = node.get_inputs().at(0).get_shape();
      Shape out_shape = node.get_output_shape(0);
      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        NGRAPH_DEBUG << "arg0_cipher->is_batched(): "
                     << arg0_cipher->is_batched();
        NGRAPH_DEBUG << "arg0_cipher->get_batch_size(): "
                     << arg0_cipher->get_batch_size();
        if (arg0_cipher->is_batched()) {
          arg0_shape[0] = arg0_shape[0] / arg0_cipher->get_batch_size();
        }

        NGRAPH_DEBUG << "out0_cipher->is_batched(): "
                     << out0_cipher->is_batched();
        NGRAPH_DEBUG << "arg0_cipher->get_batch_size(): "
                     << out0_cipher->get_batch_size();
        if (out0_cipher->is_batched()) {
          out_shape[0] = out_shape[0] / out0_cipher->get_batch_size();
        }
      }

      NGRAPH_DEBUG << "arg0_shape after batching: " << join(arg0_shape);
      NGRAPH_DEBUG << "out_shape after batching: " << join(out_shape);

      if (arg0_cipher != nullptr && arg1_cipher != nullptr &&
          out0_cipher != nullptr) {
        runtime::he::kernel::pad(
            arg0_cipher->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), arg0_shape, out_shape,
            pad->get_padding_below(), pad->get_padding_above(),
            pad->get_padding_interior(), batch_size, this);
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::pad(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), arg0_shape, out_shape,
            pad->get_padding_below(), pad->get_padding_above(),
            pad->get_padding_interior(), batch_size, this);
      } else {
        throw ngraph_error("Pad cipher vs plain types not supported.");
      }
      break;
    }
    case OP_TYPEID::Reshape: {
      NGRAPH_INFO << "Reshape op";
      const op::Reshape* reshape = static_cast<const op::Reshape*>(&node);
      if (arg0_cipher != nullptr && out0_cipher != nullptr) {
        runtime::he::kernel::reshape(
            arg0_cipher->get_elements(), out0_cipher->get_elements(),
            arg0_cipher->get_shape(), reshape->get_input_order(),
            out0_cipher->get_shape());
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        runtime::he::kernel::reshape(
            arg0_plain->get_elements(), out0_plain->get_elements(),
            arg0_plain->get_shape(), reshape->get_input_order(),
            out0_plain->get_shape());
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
                                    this);
      } else if (arg0_cipher != nullptr && out0_plain != nullptr) {
        runtime::he::kernel::result(arg0_cipher->get_elements(),
                                    out0_plain->get_elements(), output_size,
                                    this);
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        runtime::he::kernel::result(arg0_plain->get_elements(),
                                    out0_plain->get_elements(), output_size);
      } else {
        throw ngraph_error("Result types not supported.");
      }
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
            out0_cipher->get_elements(), element_type, this,
            out0_cipher->get_batched_element_count());
      } else if (arg0_cipher != nullptr && arg1_plain != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::subtract(
            arg0_cipher->get_elements(), arg1_plain->get_elements(),
            out0_cipher->get_elements(), element_type, this,
            out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && arg1_cipher != nullptr &&
                 out0_cipher != nullptr) {
        runtime::he::kernel::subtract(
            arg0_plain->get_elements(), arg1_cipher->get_elements(),
            out0_cipher->get_elements(), element_type, this,
            out0_cipher->get_batched_element_count());
      } else if (arg0_plain != nullptr && arg1_plain != nullptr &&
                 out0_plain != nullptr) {
        runtime::he::kernel::subtract(
            arg0_plain->get_elements(), arg1_plain->get_elements(),
            out0_plain->get_elements(), element_type, this,
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
            out_shape, sum->get_reduction_axes(), element_type, this);
      } else if (arg0_plain != nullptr && out0_plain != nullptr) {
        runtime::he::kernel::sum(
            arg0_plain->get_elements(), out0_plain->get_elements(), in_shape,
            out_shape, sum->get_reduction_axes(), element_type, this);
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
    case OP_TYPEID::BatchNormInference:
    case OP_TYPEID::BatchNormTraining:
    case OP_TYPEID::BatchNormTrainingBackprop:
    case OP_TYPEID::Ceiling:
    case OP_TYPEID::Convert:
    case OP_TYPEID::ConvolutionBackpropData:
    case OP_TYPEID::ConvolutionBackpropFilters:
    case OP_TYPEID::Cos:
    case OP_TYPEID::Cosh:
    case OP_TYPEID::Dequantize:
    case OP_TYPEID::Divide:
    case OP_TYPEID::EmbeddingLookup:
    case OP_TYPEID::Equal:
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
    case OP_TYPEID::MaxPool:
    case OP_TYPEID::MaxPoolBackprop:
    case OP_TYPEID::Min:
    case OP_TYPEID::Minimum:
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
    case OP_TYPEID::QuantizedMaxPool:
    case OP_TYPEID::Relu:
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
    default:
      throw unsupported_op("Unsupported op '" + node.description() + "'");
#pragma GCC diagnostic pop
  }
}