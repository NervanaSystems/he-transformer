/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sum.hpp"

#include "he_backend.hpp"
#include "he_call_frame.hpp"
#include "he_cipher_tensor_view.hpp"
#include "he_plain_tensor_view.hpp"
#include "he_seal_backend.hpp"
#include "he_tensor_view.hpp"
#include "int_call_frame.hpp"
#include "kernel/add.hpp"
#include "kernel/avg_pool.hpp"
#include "kernel/broadcast.hpp"
#include "kernel/concat.hpp"
#include "kernel/constant.hpp"
#include "kernel/convolution.hpp"
#include "kernel/dot.hpp"
#include "kernel/multiply.hpp"
#include "kernel/negate.hpp"
#include "kernel/one_hot.hpp"
#include "kernel/pad.hpp"
#include "kernel/reshape.hpp"
#include "kernel/result.hpp"
#include "kernel/slice.hpp"
#include "kernel/subtract.hpp"
#include "kernel/sum.hpp"
#include "ngraph/descriptor/layout/tensor_view_layout.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/performance_counter.hpp"
#include "ngraph/runtime/tensor_view.hpp"

using namespace std;
using namespace ngraph;

runtime::he::HECallFrame::HECallFrame(const shared_ptr<Function>& func,
                                      const shared_ptr<HEBackend>& he_backend)
    : m_function(func)
    , m_he_backend(he_backend)
{
}

bool runtime::he::HECallFrame::is_cpu_check_enabled(const shared_ptr<Node>& op) const
{
    static unordered_set<string> cpu_check_enabled_ops{
        "Sum", "Add", "Dot", "Multiply", "Convolution", "AvgPool"};
    return cpu_check_enabled_ops.count(op->description()) != 0;
}

void runtime::he::HECallFrame::call(shared_ptr<Function> function,
                                    const vector<shared_ptr<runtime::he::HETensorView>>& output_tvs,
                                    const vector<shared_ptr<runtime::he::HETensorView>>& input_tvs)
{
    // TODO: we clear timer at each run for now
    m_timer_map.clear();

    // Every descriptor::tv (inputs/outputs/intermediates) maps to one runtime::tv
    unordered_map<descriptor::TensorView*, shared_ptr<runtime::he::HETensorView>> tensor_map;

    // Map inuput descriptor::tv to runtime::tv
    size_t arg_index = 0;
    for (shared_ptr<op::Parameter> param : function->get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            descriptor::TensorView* tv = param->get_output_tensor_view(i).get();
            tensor_map.insert({tv, input_tvs[arg_index++]});
        }
    }

    // Map output descriptor::tv to runtime::tv
    for (size_t i = 0; i < function->get_output_size(); i++)
    {
        auto output_op = function->get_output_op(i);
        if (!dynamic_pointer_cast<op::Result>(output_op))
        {
            throw ngraph_error("One of function's outputs isn't op::Result");
        }
        descriptor::TensorView* tv = function->get_output_op(i)->get_output_tensor_view(0).get();
        tensor_map.insert({tv, output_tvs[i]});
    }

    // Invoke computation
    for (shared_ptr<Node> op : function->get_ordered_ops())
    {
        NGRAPH_INFO << "\033[1;32m"
                    << "[ " << op->get_name() << " ]"
                    << "\033[0m";
        if (op->description() == "Parameter")
        {
            continue;
        }
        m_timer_map[op].start();

        // Collect input runtime::tv
        vector<shared_ptr<runtime::he::HETensorView>> inputs;
        for (const descriptor::Input& input : op->get_inputs())
        {
            descriptor::TensorView* tv = input.get_output().get_tensor_view().get();
            inputs.push_back(tensor_map.at(tv));
        }

        // Collect output runtime::tv
        bool any_batched = false;
        vector<shared_ptr<runtime::he::HETensorView>> outputs;
        for (size_t i = 0; i < op->get_output_size(); ++i)
        {
            descriptor::TensorView* tv = op->get_output_tensor_view(i).get();
            string name = tv->get_tensor().get_name();
            if (!contains_key(tensor_map, tv))
            {
                // The output tensor is not in the tensor map so create a new tensor
                const Shape& shape = op->get_output_shape(i);
                const element::Type& element_type = op->get_output_element_type(i);
                string tensor_name = op->get_output_tensor(i).get_name();

                bool plain_out = all_of(
                    inputs.begin(), inputs.end(), [](shared_ptr<runtime::he::HETensorView> input) {
                        return dynamic_pointer_cast<HEPlainTensorView>(input) != nullptr;
                    });
                if (plain_out)
                {
                    auto otv = make_shared<runtime::he::HEPlainTensorView>(
                        element_type, shape, m_he_backend, name);
                    tensor_map.insert({tv, otv});
                }
                else
                {
                    bool batched_out =
                        any_of(inputs.begin(),
                               inputs.end(),
                               [](shared_ptr<runtime::he::HETensorView> input) {
                                   if (auto input_cipher_tv =
                                           dynamic_pointer_cast<HECipherTensorView>(input))
                                   {
                                       return input_cipher_tv->is_batched();
                                   }
                                   else
                                   {
                                       return false;
                                   }
                               });
                    any_batched |= batched_out;

                    auto otv = make_shared<runtime::he::HECipherTensorView>(
                        element_type, shape, m_he_backend, batched_out, name);
                    tensor_map.insert({tv, otv});
                }
            }
            outputs.push_back(tensor_map.at(tv));
        }

        element::Type base_type;
        if (op->get_inputs().empty())
        {
            base_type = op->get_element_type();
        }
        else
        {
            base_type = op->get_inputs().at(0).get_tensor().get_element_type();
        }

        generate_calls(base_type, op, outputs, inputs);

        const string op_name = op->description();

        // Check result with CPU backend
        if (is_cpu_check_enabled(op) && !any_batched)
        {
            check_cpu_calls(function, base_type, op, outputs, inputs, false);
        }

        // Check noise budget after each op
        if (auto he_seal_backend =
                dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(m_he_backend))
        {
            if (auto output = dynamic_pointer_cast<runtime::he::HECipherTensorView>(outputs[0]))
            {
                he_seal_backend->check_noise_budget(outputs);
            }
        }

        // Delete any obsolete tensors
        for (const descriptor::Tensor* t : op->liveness_free_list)
        {
            for (auto it = tensor_map.begin(); it != tensor_map.end(); ++it)
            {
                if (it->second->get_tensor().get_name() == t->get_name())
                {
                    tensor_map.erase(it);
                    break;
                }
            }
        }

        // Stop stopwatch and print time
        // TODO: currently timer is cleared at each run
        m_timer_map.at(op).stop();

        NGRAPH_INFO << "\033[1;31m" << op->get_name() << " took "
                    << m_timer_map.at(op).get_seconds() << "s"
                    << "\033[0m";
    }
}

void runtime::he::HECallFrame::check_cpu_calls(
    shared_ptr<Function> function,
    const element::Type& type,
    const shared_ptr<Node>& op,
    const vector<shared_ptr<runtime::he::HETensorView>>& outputs,
    const vector<shared_ptr<runtime::he::HETensorView>>& inputs,
    bool verbose)
{
    runtime::interpreter::INTCallFrame cpu_call_frame(function);
    vector<shared_ptr<runtime::HostTensorView>> cpu_inputs;
    vector<shared_ptr<runtime::HostTensorView>> cpu_outputs;
    vector<shared_ptr<runtime::HostTensorView>> result_outputs;

    for (shared_ptr<runtime::he::HETensorView> he_tv : inputs)
    {
        shared_ptr<HECipherTensorView> cipher_tv =
            dynamic_pointer_cast<runtime::he::HECipherTensorView>(he_tv);
        shared_ptr<HEPlainTensorView> plain_tv =
            dynamic_pointer_cast<runtime::he::HEPlainTensorView>(he_tv);

        const element::Type& type = he_tv->get_tensor_view_layout()->get_element_type();
        if (cipher_tv != nullptr)
        {
            auto shape = cipher_tv->get_expanded_shape();
            size_t num_bytes = type.size() * shape_size(shape);

            shared_ptr<HostTensorView> tv = make_shared<HostTensorView>(type, shape);
            cipher_tv->read(tv->get_data_ptr(), 0, num_bytes);
            cpu_inputs.push_back(tv);
        }
        else if (plain_tv != nullptr)
        {
            auto shape = he_tv->get_shape();
            size_t num_bytes = type.size() * shape_size(shape);
            shared_ptr<HostTensorView> tv = make_shared<HostTensorView>(type, shape);
            plain_tv->read(tv->get_data_ptr(), 0, num_bytes);
            cpu_inputs.push_back(tv);
        }
        else
        {
            throw ngraph_error("Input neither plain nor cipher tensorview.");
        }
    }

    for (shared_ptr<runtime::he::HETensorView> he_tv : outputs)
    {
        shared_ptr<HECipherTensorView> cipher_tv =
            dynamic_pointer_cast<runtime::he::HECipherTensorView>(he_tv);
        shared_ptr<HEPlainTensorView> plain_tv =
            dynamic_pointer_cast<runtime::he::HEPlainTensorView>(he_tv);

        const element::Type& type = he_tv->get_tensor_view_layout()->get_element_type();

        if (cipher_tv != nullptr)
        {
            auto shape = cipher_tv->get_expanded_shape();
            size_t num_bytes = type.size() * shape_size(shape);

            shared_ptr<HostTensorView> tv = make_shared<HostTensorView>(type, shape);
            cipher_tv->read(tv->get_data_ptr(), 0, num_bytes);
            cpu_outputs.push_back(tv);
        }
        else if (plain_tv != nullptr)
        {
            auto shape = he_tv->get_shape();
            size_t num_bytes = type.size() * shape_size(shape);
            shared_ptr<HostTensorView> tv = make_shared<HostTensorView>(type, shape);
            plain_tv->read(tv->get_data_ptr(), 0, num_bytes);
            cpu_outputs.push_back(tv);
        }
        else
        {
            throw ngraph_error("Input neither plain nor cipher tensorview.");
        }
    }
    NGRAPH_INFO << "Generating CPU calls";
    cpu_call_frame.generate_calls(type, *op, cpu_outputs, cpu_inputs);
    NGRAPH_INFO << "Generated CPU calls\n";
    const string type_name = type.c_type_string();

    // Compare outputs with CPU outputs
    bool correct = true;
    for (size_t output_ind = 0; output_ind < outputs.size(); ++output_ind)
    {
        shared_ptr<runtime::he::HETensorView> he_out = outputs[output_ind];
        shared_ptr<runtime::HostTensorView> cpu_out = cpu_outputs[output_ind];

        const element::Type& type = he_out->get_tensor_view_layout()->get_element_type();
        auto shape = cpu_out->get_shape();
        size_t num_bytes = type.size() * shape_size(shape);

        size_t element_count = cpu_out->get_element_count();
        if (type_name == "float")
        {
            vector<float> cpu_out_vec(element_count, 0);
            vector<float> he_out_vec(element_count, 0);

            he_out->read(&he_out_vec[0], 0, num_bytes);
            cpu_out->read(&cpu_out_vec[0], 0, num_bytes);

            size_t inaccurate_cnt = 0;
            for (size_t elem = 0; elem < element_count; ++elem)
            {
                if (abs(cpu_out_vec[elem] - he_out_vec[elem]) > 1e-3) // TODO: increase precision
                {
                    if (inaccurate_cnt < 10)
                    {
                        NGRAPH_INFO << "element " << elem << ": expect " << cpu_out_vec[elem]
                                    << ", actual: " << he_out_vec[elem];
                    }
                    correct = false;
                    inaccurate_cnt++;
                }
            }
            NGRAPH_INFO << inaccurate_cnt << "/" << element_count
                        << " computations are inaccurate.";
        }
        else if (type_name == "int64_t")
        {
            vector<int64_t> cpu_out_vec(element_count, 0);
            vector<int64_t> he_out_vec(element_count, 0);

            he_out->read(&he_out_vec[0], 0, num_bytes);
            cpu_out->read(&cpu_out_vec[0], 0, num_bytes);

            for (size_t elem = 0; elem < element_count; ++elem)
            {
                if (cpu_out_vec[elem] != he_out_vec[elem])
                {
                    NGRAPH_INFO << "expect " << cpu_out_vec[elem]
                                << ", actual: " << he_out_vec[elem];
                    correct = false;
                }
            }
        }
        else
        {
            throw ngraph_error("CPU checking for type " + type_name + " not enabled");
        }
    }
    if (!correct || verbose)
    {
        if (!verbose)
        {
            NGRAPH_INFO << "Inaccurate float computation.";
        }
        else
        {
            NGRAPH_INFO << "Verbose float computation";
        }

        auto print_tensor_view = [&type](shared_ptr<runtime::HostTensorView> tv) -> void {
            size_t element_count = tv->get_element_count();
            auto shape = tv->get_shape();
            size_t num_bytes = type.size() * shape_size(shape);
            vector<float> tv_vec(element_count, 0);
            tv->read(&tv_vec[0], 0, num_bytes);
            for (auto elem : tv_vec)
            {
                cout << elem << " ";
            }
            cout << endl;
        };
        /* for (shared_ptr<runtime::HostTensorView> cpu_input : cpu_inputs)
        {
            NGRAPH_INFO << "Input";
            print_tensor_view(cpu_input);
        }
        for (shared_ptr<runtime::HostTensorView> cpu_output : cpu_outputs)
        {
            NGRAPH_INFO << "Output";
            print_tensor_view(cpu_output);
        } */
        if (!correct)
        {
            NGRAPH_INFO << "Inaccurate float computation";
            throw ngraph_error("Inaccurate float computation");
        }
    }
    NGRAPH_INFO << "HE op matches CPU call";
}

void runtime::he::HECallFrame::generate_calls(const element::Type& type,
                                              const shared_ptr<Node>& node,
                                              const vector<shared_ptr<HETensorView>>& out,
                                              const vector<shared_ptr<HETensorView>>& args)
{
    string node_op = node->description();
    shared_ptr<HECipherTensorView> arg0_cipher = nullptr;
    shared_ptr<HEPlainTensorView> arg0_plain = nullptr;
    shared_ptr<HECipherTensorView> arg1_cipher = nullptr;
    shared_ptr<HEPlainTensorView> arg1_plain = nullptr;
    shared_ptr<HECipherTensorView> out0_cipher = dynamic_pointer_cast<HECipherTensorView>(out[0]);
    shared_ptr<HEPlainTensorView> out0_plain = dynamic_pointer_cast<HEPlainTensorView>(out[0]);

    if (args.size() > 0)
    {
        arg0_cipher = dynamic_pointer_cast<HECipherTensorView>(args[0]);
        arg0_plain = dynamic_pointer_cast<HEPlainTensorView>(args[0]);
    }
    if (args.size() > 1)
    {
        arg1_cipher = dynamic_pointer_cast<HECipherTensorView>(args[1]);
        arg1_plain = dynamic_pointer_cast<HEPlainTensorView>(args[1]);
    }

    size_t batch_size = 1;
    if (out0_cipher != nullptr)
    {
        batch_size = out0_cipher->get_batch_size();
    }

    if (node_op == "Add")
    {
        if (arg0_cipher != nullptr && arg1_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::add(arg0_cipher->get_elements(),
                                     arg1_cipher->get_elements(),
                                     out0_cipher->get_elements(),
                                     type,
                                     m_he_backend,
                                     out0_cipher->get_element_count());
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::add(arg0_cipher->get_elements(),
                                     arg1_plain->get_elements(),
                                     out0_cipher->get_elements(),
                                     type,
                                     m_he_backend,
                                     out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::add(arg0_plain->get_elements(),
                                     arg1_cipher->get_elements(),
                                     out0_cipher->get_elements(),
                                     type,
                                     m_he_backend,
                                     out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_plain != nullptr && out0_plain != nullptr)
        {
            runtime::he::kernel::add(arg0_plain->get_elements(),
                                     arg1_plain->get_elements(),
                                     out0_plain->get_elements(),
                                     type,
                                     m_he_backend,
                                     out0_plain->get_element_count());
        }
        else
        {
            throw ngraph_error("Add types not supported.");
        }
    }
    else if (node_op == "AvgPool")
    {
        shared_ptr<op::AvgPool> avg_pool = dynamic_pointer_cast<op::AvgPool>(node);

        if (arg0_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::avg_pool(arg0_cipher->get_elements(),
                                          out0_cipher->get_elements(),
                                          arg0_cipher->get_shape(),
                                          out0_cipher->get_shape(),
                                          avg_pool->get_window_shape(),
                                          avg_pool->get_window_movement_strides(),
                                          avg_pool->get_padding_below(),
                                          avg_pool->get_padding_above(),
                                          avg_pool->get_include_padding_in_avg_computation(),
                                          type,
                                          m_he_backend);
        }
        else if (arg0_plain != nullptr && out0_plain != nullptr)
        {
            runtime::he::kernel::avg_pool(arg0_plain->get_elements(),
                                          out0_plain->get_elements(),
                                          arg0_plain->get_shape(),
                                          out0_plain->get_shape(),
                                          avg_pool->get_window_shape(),
                                          avg_pool->get_window_movement_strides(),
                                          avg_pool->get_padding_below(),
                                          avg_pool->get_padding_above(),
                                          avg_pool->get_include_padding_in_avg_computation(),
                                          type,
                                          m_he_backend);
        }
        else
        {
            throw ngraph_error("AvgPool types not supported");
        }
    }
    else if (node_op == "Broadcast")
    {
        shared_ptr<op::Broadcast> broadcast = dynamic_pointer_cast<op::Broadcast>(node);
        AxisSet broadcast_axes = broadcast->get_broadcast_axes();

        if (arg0_cipher != nullptr && out0_cipher != nullptr)
        {
            Shape in_shape = arg0_cipher->get_shape();
            Shape out_shape = out0_cipher->get_shape();
            runtime::he::kernel::broadcast(arg0_cipher->get_elements(),
                                           out0_cipher->get_elements(),
                                           in_shape,
                                           out_shape,
                                           broadcast_axes);
        }
        else if (arg0_plain != nullptr && out0_plain != nullptr)
        {
            Shape in_shape = arg0_plain->get_shape();
            Shape out_shape = out0_plain->get_shape();
            runtime::he::kernel::broadcast(arg0_plain->get_elements(),
                                           out0_plain->get_elements(),
                                           in_shape,
                                           out_shape,
                                           broadcast_axes);
        }
        else
        {
            throw ngraph_error("Broadcast types not supported.");
        }
    }
    else if (node_op == "Concat")
    {
        shared_ptr<op::Concat> concat = dynamic_pointer_cast<op::Concat>(node);
        if (arg0_cipher != nullptr && out0_cipher != nullptr)
        {
            vector<vector<shared_ptr<runtime::he::HECiphertext>>> in_args;
            vector<Shape> in_shapes;
            for (shared_ptr<HETensorView> arg : args)
            {
                shared_ptr<HECipherTensorView> arg_cipher =
                    dynamic_pointer_cast<HECipherTensorView>(arg);
                if (arg_cipher == nullptr)
                {
                    throw ngraph_error("Concat type not consistent");
                }
                in_args.push_back(arg_cipher->get_elements());
                in_shapes.push_back(arg_cipher->get_shape());

                runtime::he::kernel::concat(in_args,
                                            out0_cipher->get_elements(),
                                            in_shapes,
                                            out0_cipher->get_shape(),
                                            concat->get_concatenation_axis());
            }
        }
        else if (arg0_plain != nullptr && out0_plain != nullptr)
        {
            vector<vector<shared_ptr<runtime::he::HEPlaintext>>> in_args;
            vector<Shape> in_shapes;
            for (shared_ptr<HETensorView> arg : args)
            {
                shared_ptr<HEPlainTensorView> arg_plain =
                    dynamic_pointer_cast<HEPlainTensorView>(arg);
                if (arg_plain == nullptr)
                {
                    throw ngraph_error("Concat type not consistent");
                }
                in_args.push_back(arg_plain->get_elements());
                in_shapes.push_back(arg_plain->get_shape());

                runtime::he::kernel::concat(in_args,
                                            out0_plain->get_elements(),
                                            in_shapes,
                                            out0_plain->get_shape(),
                                            concat->get_concatenation_axis());
            }
        }
        else
        {
            throw ngraph_error("Concat types not supported.");
        }
    }
    else if (node_op == "Constant")
    {
        if (out0_plain != nullptr)
        {
            shared_ptr<op::Constant> constant = static_pointer_cast<op::Constant>(node);
            runtime::he::kernel::constant(out0_plain->get_elements(),
                                          type,
                                          constant->get_data_ptr(),
                                          m_he_backend,
                                          out0_plain->get_element_count());
        }
        else
        {
            throw ngraph_error("Constant type not supported.");
        }
    }
    else if (node_op == "Convolution")
    {
        shared_ptr<op::Convolution> c = dynamic_pointer_cast<op::Convolution>(node);

        if (arg0_cipher != nullptr && arg1_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::convolution(arg0_cipher->get_elements(),
                                             arg1_cipher->get_elements(),
                                             out0_cipher->get_elements(),
                                             arg0_cipher->get_shape(),
                                             arg1_cipher->get_shape(),
                                             out0_cipher->get_shape(),
                                             c->get_window_movement_strides(),
                                             c->get_window_dilation_strides(),
                                             c->get_padding_below(),
                                             c->get_padding_above(),
                                             c->get_data_dilation_strides(),
                                             0,
                                             1,
                                             1,
                                             0,
                                             0,
                                             1,
                                             false,
                                             type,
                                             batch_size,
                                             m_he_backend);
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::convolution(arg0_cipher->get_elements(),
                                             arg1_plain->get_elements(),
                                             out0_cipher->get_elements(),
                                             arg0_cipher->get_shape(),
                                             arg1_plain->get_shape(),
                                             out0_cipher->get_shape(),
                                             c->get_window_movement_strides(),
                                             c->get_window_dilation_strides(),
                                             c->get_padding_below(),
                                             c->get_padding_above(),
                                             c->get_data_dilation_strides(),
                                             0,
                                             1,
                                             1,
                                             0,
                                             0,
                                             1,
                                             false,
                                             type,
                                             batch_size,
                                             m_he_backend);
        }
        else if (arg0_plain != nullptr && arg1_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::convolution(arg0_plain->get_elements(),
                                             arg1_cipher->get_elements(),
                                             out0_cipher->get_elements(),
                                             arg0_plain->get_shape(),
                                             arg1_cipher->get_shape(),
                                             out0_cipher->get_shape(),
                                             c->get_window_movement_strides(),
                                             c->get_window_dilation_strides(),
                                             c->get_padding_below(),
                                             c->get_padding_above(),
                                             c->get_data_dilation_strides(),
                                             0,
                                             1,
                                             1,
                                             0,
                                             0,
                                             1,
                                             false,
                                             type,
                                             batch_size,
                                             m_he_backend);
        }
        else if (arg0_plain != nullptr && arg1_plain != nullptr && out0_plain != nullptr)
        {
            runtime::he::kernel::convolution(arg0_plain->get_elements(),
                                             arg1_plain->get_elements(),
                                             out0_plain->get_elements(),
                                             arg0_plain->get_shape(),
                                             arg1_plain->get_shape(),
                                             out0_plain->get_shape(),
                                             c->get_window_movement_strides(),
                                             c->get_window_dilation_strides(),
                                             c->get_padding_below(),
                                             c->get_padding_above(),
                                             c->get_data_dilation_strides(),
                                             0,
                                             1,
                                             1,
                                             0,
                                             0,
                                             1,
                                             false,
                                             type,
                                             batch_size,
                                             m_he_backend);
        }
        else
        {
            throw ngraph_error("Convolution types not supported.");
        }
    }
    else if (node_op == "Dot")
    {
        shared_ptr<op::Dot> dot = dynamic_pointer_cast<op::Dot>(node);
        NGRAPH_INFO << join(args[0]->get_shape(), "x") << " dot "
                    << join(args[1]->get_shape(), "x");

        if (arg0_cipher != nullptr && arg1_cipher != nullptr && out0_cipher != nullptr)
        {
            NGRAPH_INFO << "Dot cipher cipher => cipher";
            runtime::he::kernel::dot(arg0_cipher->get_elements(),
                                     arg1_cipher->get_elements(),
                                     out0_cipher->get_elements(),
                                     arg0_cipher->get_shape(),
                                     arg1_cipher->get_shape(),
                                     out0_cipher->get_shape(),
                                     dot->get_reduction_axes_count(),
                                     type,
                                     batch_size,
                                     m_he_backend);
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr && out0_cipher != nullptr)
        {
            NGRAPH_INFO << "Dot cipher plain => cipher";
            runtime::he::kernel::dot(arg0_cipher->get_elements(),
                                     arg1_plain->get_elements(),
                                     out0_cipher->get_elements(),
                                     arg0_cipher->get_shape(),
                                     arg1_plain->get_shape(),
                                     out0_cipher->get_shape(),
                                     dot->get_reduction_axes_count(),
                                     type,
                                     batch_size,
                                     m_he_backend);
        }
        else if (arg0_plain != nullptr && arg1_cipher != nullptr && out0_cipher != nullptr)
        {
            NGRAPH_INFO << "Dot plain cipher => cipher";
            runtime::he::kernel::dot(arg0_plain->get_elements(),
                                     arg1_cipher->get_elements(),
                                     out0_cipher->get_elements(),
                                     arg0_plain->get_shape(),
                                     arg1_cipher->get_shape(),
                                     out0_cipher->get_shape(),
                                     dot->get_reduction_axes_count(),
                                     type,
                                     batch_size,
                                     m_he_backend);
        }
        else if (arg0_plain != nullptr && arg1_plain != nullptr && out0_plain != nullptr)
        {
            NGRAPH_INFO << "Dot plain plain => plain";
            runtime::he::kernel::dot(arg0_plain->get_elements(),
                                     arg1_plain->get_elements(),
                                     out0_plain->get_elements(),
                                     arg0_plain->get_shape(),
                                     arg1_plain->get_shape(),
                                     out0_plain->get_shape(),
                                     dot->get_reduction_axes_count(),
                                     type,
                                     m_he_backend);
        }
        else
        {
            throw ngraph_error("Dot types not supported.");
        }
    }
    else if (node_op == "Multiply")
    {
        if (arg0_cipher != nullptr && arg1_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::multiply(arg0_cipher->get_elements(),
                                          arg1_cipher->get_elements(),
                                          out0_cipher->get_elements(),
                                          type,
                                          m_he_backend,
                                          out0_cipher->get_element_count());
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::multiply(arg0_cipher->get_elements(),
                                          arg1_plain->get_elements(),
                                          out0_cipher->get_elements(),
                                          type,
                                          m_he_backend,
                                          out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::multiply(arg0_plain->get_elements(),
                                          arg1_cipher->get_elements(),
                                          out0_cipher->get_elements(),
                                          type,
                                          m_he_backend,
                                          out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_plain != nullptr && out0_plain != nullptr)
        {
            runtime::he::kernel::multiply(arg0_plain->get_elements(),
                                          arg1_plain->get_elements(),
                                          out0_plain->get_elements(),
                                          type,
                                          m_he_backend,
                                          out0_plain->get_element_count());
        }
        else
        {
            throw ngraph_error("Multiply types not supported.");
        }
    }
    else if (node_op == "Negative")
    {
        if (arg0_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::negate(arg0_cipher->get_elements(),
                                        out0_cipher->get_elements(),
                                        type,
                                        m_he_backend,
                                        out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && out0_plain != nullptr)
        {
            runtime::he::kernel::negate(arg0_plain->get_elements(),
                                        out0_plain->get_elements(),
                                        type,
                                        m_he_backend,
                                        out0_plain->get_element_count());
        }
        else
        {
            throw ngraph_error("Negate types not supported.");
        }
    }
    else if (node_op == "OneHot")
    {
        shared_ptr<op::OneHot> oh = dynamic_pointer_cast<op::OneHot>(node);

        if (arg0_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::one_hot(arg0_cipher->get_elements(),
                                         out0_cipher->get_elements(),
                                         arg0_cipher->get_shape(),
                                         out0_cipher->get_shape(),
                                         oh->get_one_hot_axis(),
                                         type,
                                         m_he_backend);
        }
        else
        {
            throw ngraph_error("OneHot types not supported.");
        }
    }
    else if (node_op == "Reshape")
    {
        shared_ptr<op::Reshape> reshape = dynamic_pointer_cast<op::Reshape>(node);

        if (arg0_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::reshape(arg0_cipher->get_elements(),
                                         out0_cipher->get_elements(),
                                         arg0_cipher->get_shape(),
                                         reshape->get_input_order(),
                                         out0_cipher->get_shape());
        }
        else if (arg0_plain != nullptr && out0_plain != nullptr)
        {
            runtime::he::kernel::reshape(arg0_plain->get_elements(),
                                         out0_plain->get_elements(),
                                         arg0_plain->get_shape(),
                                         reshape->get_input_order(),
                                         out0_plain->get_shape());
        }
        else
        {
            throw ngraph_error("Reshape types not supported.");
        }
    }
    else if (node_op == "Result")
    {
        shared_ptr<op::Result> res = dynamic_pointer_cast<op::Result>(node);

        if (arg0_cipher != nullptr && out0_cipher != nullptr)
        {
            size_t output_size = shape_size(res->get_shape());
            if (arg0_cipher->is_batched())
            {
                output_size /= arg0_cipher->get_batch_size();
            }
            runtime::he::kernel::result(
                arg0_cipher->get_elements(), out0_cipher->get_elements(), output_size);
        }
        else if (arg0_plain != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::result(arg0_plain->get_elements(),
                                        out0_cipher->get_elements(),
                                        shape_size(res->get_shape()),
                                        m_he_backend);
        }
        else if (arg0_plain != nullptr && out0_plain != nullptr)
        {
            runtime::he::kernel::result(arg0_plain->get_elements(),
                                        out0_plain->get_elements(),
                                        shape_size(res->get_shape()));
        }
        else
        {
            throw ngraph_error("Result types not supported.");
        }
    }
    else if (node_op == "Slice")
    {
        shared_ptr<op::Slice> slice = dynamic_pointer_cast<op::Slice>(node);
        if (arg0_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::slice(arg0_cipher->get_elements(),
                                       out0_cipher->get_elements(),
                                       arg0_cipher->get_shape(),
                                       slice->get_lower_bounds(),
                                       slice->get_upper_bounds(),
                                       slice->get_strides(),
                                       out0_cipher->get_shape());
        }
        else if (arg0_plain != nullptr && out0_plain != nullptr)
        {
            runtime::he::kernel::slice(arg0_plain->get_elements(),
                                       out0_plain->get_elements(),
                                       arg0_plain->get_shape(),
                                       slice->get_lower_bounds(),
                                       slice->get_upper_bounds(),
                                       slice->get_strides(),
                                       out0_plain->get_shape());
        }
        else
        {
            throw ngraph_error("Slice types not supported.");
        }
    }
    else if (node_op == "Subtract")
    {
        if (arg0_cipher != nullptr && arg1_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::subtract(arg0_cipher->get_elements(),
                                          arg1_cipher->get_elements(),
                                          out0_cipher->get_elements(),
                                          type,
                                          m_he_backend,
                                          out0_cipher->get_element_count());
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::subtract(arg0_cipher->get_elements(),
                                          arg1_plain->get_elements(),
                                          out0_cipher->get_elements(),
                                          type,
                                          m_he_backend,
                                          out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::subtract(arg0_plain->get_elements(),
                                          arg1_cipher->get_elements(),
                                          out0_cipher->get_elements(),
                                          type,
                                          m_he_backend,
                                          out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_plain != nullptr && out0_plain != nullptr)
        {
            runtime::he::kernel::subtract(arg0_plain->get_elements(),
                                          arg1_plain->get_elements(),
                                          out0_plain->get_elements(),
                                          type,
                                          m_he_backend,
                                          out0_plain->get_element_count());
        }
        else
        {
            throw ngraph_error("Subtract types not supported.");
        }
    }
    else if (node_op == "Sum")
    {
        shared_ptr<op::Sum> sum = dynamic_pointer_cast<op::Sum>(node);
        if (arg0_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::sum(arg0_cipher->get_elements(),
                                     out0_cipher->get_elements(),
                                     arg0_cipher->get_shape(),
                                     out0_cipher->get_shape(),
                                     sum->get_reduction_axes(),
                                     type,
                                     m_he_backend);
        }
        else if (arg0_plain != nullptr && out0_plain != nullptr)
        {
            runtime::he::kernel::sum(arg0_plain->get_elements(),
                                     out0_plain->get_elements(),
                                     arg0_plain->get_shape(),
                                     out0_plain->get_shape(),
                                     sum->get_reduction_axes(),
                                     type,
                                     m_he_backend);
        }
        else
        {
            throw ngraph_error("Sum types not supported.");
        }
    }
    else if (node_op == "Pad")
    {
        shared_ptr<op::Pad> pad = dynamic_pointer_cast<op::Pad>(node);

        if (arg0_cipher != nullptr && arg1_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::pad(arg0_cipher->get_elements(),
                                     arg1_cipher->get_elements(),
                                     out0_cipher->get_elements(),
                                     node->get_inputs().at(0).get_shape(),
                                     node->get_output_shape(0),
                                     pad->get_padding_below(),
                                     pad->get_padding_above(),
                                     pad->get_padding_interior(),
                                     m_he_backend);
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::pad(arg0_cipher->get_elements(),
                                     arg1_plain->get_elements(),
                                     out0_cipher->get_elements(),
                                     node->get_inputs().at(0).get_shape(),
                                     node->get_output_shape(0),
                                     pad->get_padding_below(),
                                     pad->get_padding_above(),
                                     pad->get_padding_interior(),
                                     m_he_backend);
        }
        else
        {
            throw ngraph_error("Pad cipher vs plain types not supported.");
        }
    }
    else
    {
        throw ngraph_error("Node op " + node_op + " unimplemented");
    }
}

void runtime::he::HECallFrame::call(const vector<shared_ptr<runtime::TensorView>>& output_tvs,
                                    const vector<shared_ptr<runtime::TensorView>>& input_tvs)
{
    vector<shared_ptr<runtime::he::HETensorView>> args;
    vector<shared_ptr<runtime::he::HETensorView>> out;
    for (auto tv : input_tvs)
    {
        args.push_back(static_pointer_cast<runtime::he::HETensorView>(tv));
    }
    for (auto tv : output_tvs)
    {
        out.push_back(static_pointer_cast<runtime::he::HETensorView>(tv));
    }
    call(m_function, out, args);
}

vector<runtime::PerformanceCounter> runtime::he::HECallFrame::get_performance_data() const
{
    vector<runtime::PerformanceCounter> rc;
    for (pair<shared_ptr<Node>, stopwatch> p : m_timer_map)
    {
        rc.emplace_back(p.first->get_name().c_str(),
                        p.second.get_total_microseconds(),
                        p.second.get_call_count());
    }
    return rc;
}
