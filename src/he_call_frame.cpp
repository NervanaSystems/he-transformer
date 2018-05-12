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
#include "ngraph/op/dot.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sum.hpp"

#include "he_backend.hpp"
#include "he_call_frame.hpp"
#include "he_cipher_tensor_view.hpp"
#include "he_plain_tensor_view.hpp"
#include "he_tensor_view.hpp"
#include "kernel/add.hpp"
#include "kernel/broadcast.hpp"
#include "kernel/concat.hpp"
#include "kernel/constant.hpp"
#include "kernel/dot.hpp"
#include "kernel/multiply.hpp"
#include "kernel/one_hot.hpp"
#include "kernel/relinearize.hpp"
#include "kernel/reshape.hpp"
#include "kernel/result.hpp"
#include "kernel/slice.hpp"
#include "kernel/subtract.hpp"
#include "kernel/sum.hpp"
#include "ngraph/descriptor/layout/tensor_view_layout.hpp"
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
                if (op->description() == "Constant") // Always result in plaintext
                {
                    auto itv = make_shared<runtime::he::HEPlainTensorView>(
                        element_type, shape, m_he_backend, name);
                    tensor_map.insert({tv, itv});
                } // one-input ops that prefer plaintext result
                else if (op->description() == "Broadcast" || op->description() == "Reshape")
                {
                    shared_ptr<HEPlainTensorView> in0_plain =
                        dynamic_pointer_cast<HEPlainTensorView>(inputs[0]);
                    if (in0_plain != nullptr)
                    {
                        auto itv = make_shared<runtime::he::HEPlainTensorView>(
                            element_type, shape, m_he_backend, name);
                        tensor_map.insert({tv, itv});
                    }
                    else
                    {
                        auto itv = make_shared<runtime::he::HECipherTensorView>(
                            element_type, shape, m_he_backend, name);
                        tensor_map.insert({tv, itv});
                    }
                } // two-input ops that prefer plaintetx result
                else if (op->description() == "Add" || op->description() == "Multiply" ||
                         op->description() == "Dot")
                {
                    shared_ptr<HEPlainTensorView> in0_plain =
                        dynamic_pointer_cast<HEPlainTensorView>(inputs[0]);
                    shared_ptr<HEPlainTensorView> in1_plain =
                        dynamic_pointer_cast<HEPlainTensorView>(inputs[1]);
                    if ((in0_plain != nullptr) && (in1_plain != nullptr))
                    {
                        NGRAPH_INFO << "Op " << op->description() << " out is plaintext";
                        auto itv = make_shared<runtime::he::HEPlainTensorView>(
                            element_type, shape, m_he_backend, name);
                        tensor_map.insert({tv, itv});
                    }
                    else
                    {
                        auto itv = make_shared<runtime::he::HECipherTensorView>(
                            element_type, shape, m_he_backend, name);
                        tensor_map.insert({tv, itv});
                    }
                }
                else
                {
                    auto itv = make_shared<runtime::he::HECipherTensorView>(
                        element_type, shape, m_he_backend, name);
                    tensor_map.insert({tv, itv});
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

        generate_calls(base_type, op, inputs, outputs);

        const string op_name = op->description();
        static unordered_set<string> cpu_check_enabled_ops{"Sum", "Add", "Dot", "Multiply"};
        bool cpu_check = cpu_check_enabled_ops.count(op_name) != 0;

        if (cpu_check)
        {
            check_cpu_calls(function, base_type, op, inputs, outputs, false);
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

        // Check noise budget after each op
        m_he_backend->check_noise_budget(outputs);

        // Stop stopwatch and print time
        // TODO: currently timer is cleared at each run
        m_timer_map.at(op).stop();

        NGRAPH_INFO << "\033[1;31m" << op->get_name() << " took "
                    << m_timer_map.at(op).get_seconds() << "s"
                    << "\033[0m";
    }

    // Check noise budget at for all function outputs
    m_he_backend->check_noise_budget(output_tvs);
}

void runtime::he::HECallFrame::check_cpu_calls(
    shared_ptr<Function> function,
    const element::Type& type,
    const shared_ptr<Node>& op,
    const vector<shared_ptr<runtime::he::HETensorView>>& inputs,
    const vector<shared_ptr<runtime::he::HETensorView>>& outputs,
    bool verbose)
{
    runtime::interpreter::INT_CallFrame cpu_call_frame(function);
    std::vector<std::shared_ptr<runtime::HostTensorView>> cpu_inputs;
    std::vector<std::shared_ptr<runtime::HostTensorView>> cpu_outputs;
    std::vector<std::shared_ptr<runtime::HostTensorView>> result_outputs;

    for (std::shared_ptr<runtime::he::HETensorView> he_tv : inputs)
    {
        std::shared_ptr<HECipherTensorView> cipher_tv =
            dynamic_pointer_cast<runtime::he::HECipherTensorView>(he_tv);
        std::shared_ptr<HEPlainTensorView> plain_tv =
            dynamic_pointer_cast<runtime::he::HEPlainTensorView>(he_tv);

        const element::Type& type = he_tv->get_tensor_view_layout()->get_element_type();
        auto shape = he_tv->get_shape();
        size_t num_bytes = type.size() * shape_size(shape);
        shared_ptr<HostTensorView> tv = make_shared<HostTensorView>(type, shape);

        if (cipher_tv != nullptr)
        {
            cipher_tv->read(tv->get_data_ptr(), 0, num_bytes);
        }
        else if (plain_tv != nullptr)
        {
            plain_tv->read(tv->get_data_ptr(), 0, num_bytes);
        }
        else
        {
            throw ngraph_error("Input neither plain nor cipher tensorview.");
        }
        cpu_inputs.push_back(tv);
    }

    for (std::shared_ptr<runtime::he::HETensorView> he_tv : outputs)
    {
        const element::Type& type = he_tv->get_tensor_view_layout()->get_element_type();
        auto shape = he_tv->get_shape();
        size_t num_bytes = type.size() * shape_size(shape);
        shared_ptr<HostTensorView> tv = make_shared<HostTensorView>(type, shape);
        cpu_outputs.push_back(tv);
    }
    NGRAPH_INFO << "Generating CPU calls";
    cpu_call_frame.generate_calls(type, type, *op, cpu_inputs, cpu_outputs);
    const string type_name = type.c_type_string();

    // Compare outputs with CPU outputs
    bool correct = true;
    for (size_t output_ind = 0; output_ind < outputs.size(); ++output_ind)
    {
        std::shared_ptr<runtime::he::HETensorView> he_out = outputs[output_ind];
        std::shared_ptr<runtime::HostTensorView> cpu_out = cpu_outputs[output_ind];

        const element::Type& type = he_out->get_tensor_view_layout()->get_element_type();
        auto shape = he_out->get_shape();
        size_t num_bytes = type.size() * shape_size(shape);

        if (type_name == "float")
        {
            size_t element_count = he_out->get_element_count();
            vector<float> cpu_out_vec(element_count, 0);
            vector<float> he_out_vec(element_count, 0);

            he_out->read(&he_out_vec[0], 0, num_bytes);
            cpu_out->read(&cpu_out_vec[0], 0, num_bytes);

            for (size_t elem = 0; elem < element_count; ++elem)
            {
                if (abs(cpu_out_vec[elem] - he_out_vec[elem]) > 0.001)
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
        for (std::shared_ptr<runtime::HostTensorView> cpu_input : cpu_inputs)
        {
            NGRAPH_INFO << "Input";
            size_t element_count = cpu_input->get_element_count();
            auto shape = cpu_input->get_shape();
            size_t num_bytes = type.size() * shape_size(shape);
            vector<float> cpu_inp_vec(element_count, 0);
            cpu_input->read(&cpu_inp_vec[0], 0, num_bytes);
            for (auto elem : cpu_inp_vec)
            {
                cout << elem << " ";
            }
            cout << endl;
        }
        for (std::shared_ptr<runtime::HostTensorView> cpu_output : cpu_outputs)
        {
            NGRAPH_INFO << "output";
            size_t element_count = cpu_output->get_element_count();
            auto shape = cpu_output->get_shape();
            size_t num_bytes = type.size() * shape_size(shape);
            vector<float> cpu_inp_vec(element_count, 0);
            cpu_output->read(&cpu_inp_vec[0], 0, num_bytes);
            for (auto elem : cpu_inp_vec)
            {
                cout << elem << " ";
            }
            cout << endl;
        }
        if (!correct)
        {
            throw ngraph_error("Inaccurate float computation");
        }
    }
    NGRAPH_INFO << "HE op matches CPU call";
}

void runtime::he::HECallFrame::generate_calls(const element::Type& type,
                                              const shared_ptr<Node>& node,
                                              const vector<shared_ptr<HETensorView>>& args,
                                              const vector<shared_ptr<HETensorView>>& out)
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

    if (node_op == "Add")
    {
        if (arg0_cipher != nullptr && arg1_cipher != nullptr)
        {
            runtime::he::kernel::add(arg0_cipher->get_elements(),
                                     arg1_cipher->get_elements(),
                                     out0_cipher->get_elements(),
                                     m_he_backend,
                                     out0_cipher->get_element_count());
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr)
        {
            runtime::he::kernel::add(arg0_cipher->get_elements(),
                                     arg1_plain->get_elements(),
                                     out0_cipher->get_elements(),
                                     m_he_backend,
                                     out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_cipher != nullptr)
        {
            runtime::he::kernel::add(arg0_plain->get_elements(),
                                     arg1_cipher->get_elements(),
                                     out0_cipher->get_elements(),
                                     m_he_backend,
                                     out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_plain != nullptr)
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
    else if (node_op == "Broadcast")
    {
        shared_ptr<op::Broadcast> broadcast = dynamic_pointer_cast<op::Broadcast>(node);
        AxisSet broadcast_axes = broadcast->get_broadcast_axes();

        if (arg0_cipher != nullptr && out0_cipher != nullptr)
        {
            NGRAPH_INFO << "Broadcast cipher cipher ";
            Shape in_shape = arg0_cipher->get_shape();
            Shape out_shape = out0_cipher->get_shape();
            runtime::he::kernel::broadcast(arg0_cipher->get_elements(),
                                           out0_cipher->get_elements(),
                                           in_shape,
                                           out_shape,
                                           broadcast_axes);
        }
        else if (arg0_plain != nullptr && out0_cipher != nullptr)
        {
            NGRAPH_INFO << "Broadcast plain -> cipher ";
            Shape in_shape = arg0_plain->get_shape();
            Shape out_shape = out0_cipher->get_shape();
            runtime::he::kernel::broadcast(arg0_plain->get_elements(),
                                           out0_cipher->get_elements(),
                                           in_shape,
                                           out_shape,
                                           broadcast_axes,
                                           m_he_backend);
        }
        else if (arg0_plain != nullptr && out0_plain != nullptr)
        {
            NGRAPH_INFO << "Broadcast plain plain";
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
            vector<vector<shared_ptr<seal::Ciphertext>>> in_args;
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
    else if (node_op == "Dot")
    {
        shared_ptr<op::Dot> dot = dynamic_pointer_cast<op::Dot>(node);

        if (arg0_cipher != nullptr && arg1_cipher != nullptr)
        {
            NGRAPH_INFO << "Dot cipher cipher";
            runtime::he::kernel::dot(arg0_cipher->get_elements(),
                                     arg1_cipher->get_elements(),
                                     out0_cipher->get_elements(),
                                     arg0_cipher->get_shape(),
                                     arg1_cipher->get_shape(),
                                     out0_cipher->get_shape(),
                                     dot->get_reduction_axes_count(),
                                     type,
                                     m_he_backend);
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr)
        {
            NGRAPH_INFO << "Dot cipher plain";
            runtime::he::kernel::dot(arg0_cipher->get_elements(),
                                     arg1_plain->get_elements(),
                                     out0_cipher->get_elements(),
                                     arg0_cipher->get_shape(),
                                     arg1_plain->get_shape(),
                                     out0_cipher->get_shape(),
                                     dot->get_reduction_axes_count(),
                                     type,
                                     m_he_backend);
        }
        else if (arg0_plain != nullptr && arg1_cipher != nullptr)
        {
            NGRAPH_INFO << "Dot cipher plain";
            runtime::he::kernel::dot(arg0_plain->get_elements(),
                                     arg1_cipher->get_elements(),
                                     out0_cipher->get_elements(),
                                     arg0_plain->get_shape(),
                                     arg1_cipher->get_shape(),
                                     out0_cipher->get_shape(),
                                     dot->get_reduction_axes_count(),
                                     type,
                                     m_he_backend);
        }
        else if (arg0_plain != nullptr && arg1_plain != nullptr)
        {
            NGRAPH_INFO << "Dot plain plain";
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
        if (arg0_cipher != nullptr && arg1_cipher != nullptr)
        {
            runtime::he::kernel::multiply(arg0_cipher->get_elements(),
                                          arg1_cipher->get_elements(),
                                          out0_cipher->get_elements(),
                                          type,
                                          m_he_backend,
                                          out0_cipher->get_element_count());
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr)
        {
            runtime::he::kernel::multiply(arg0_cipher->get_elements(),
                                          arg1_plain->get_elements(),
                                          out0_cipher->get_elements(),
                                          type,
                                          m_he_backend,
                                          out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_cipher != nullptr)
        {
            runtime::he::kernel::multiply(arg0_plain->get_elements(),
                                          arg1_cipher->get_elements(),
                                          out0_cipher->get_elements(),
                                          type,
                                          m_he_backend,
                                          out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_plain != nullptr)
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
    else if (node_op == "Relinearize")
    {
        if (arg0_cipher != nullptr && out0_cipher != nullptr)
        {
            NGRAPH_INFO << "Relin? cipher cipehr";
            runtime::he::kernel::relinearize(arg0_cipher->get_elements(),
                                             out0_cipher->get_elements(),
                                             m_he_backend,
                                             out0_cipher->get_element_count());
        }
        else
        {
            NGRAPH_INFO << "arg0 is plaintext? " << (arg0_plain != nullptr)
                        << ", out0 is plaintext? " << (out0_plain != nullptr);
            //throw ngraph_error("Input to Relinearize must be ciphertext");
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
        else if (arg0_cipher != nullptr && out0_plain != nullptr)
        {
            NGRAPH_INFO << "arg0_cipher, out0_plain";
            throw ngraph_error("Reshape types not supported.");
        }
        else if (arg0_plain != nullptr && out0_cipher != nullptr)
        {
            NGRAPH_INFO << "arg0_plain, out0_cipher"; // TODO next
            runtime::he::kernel::reshape(arg0_plain->get_elements(),
                                         out0_cipher->get_elements(),
                                         arg0_plain->get_shape(),
                                         reshape->get_input_order(),
                                         out0_cipher->get_shape(),
                                         m_he_backend);
        }
        else if (arg0_plain != nullptr && out0_plain != nullptr)
        {
            NGRAPH_INFO << "plain plain";
            runtime::he::kernel::reshape(arg0_plain->get_elements(),
                                         out0_plain->get_elements(),
                                         arg0_plain->get_shape(),
                                         reshape->get_input_order(),
                                         out0_plain->get_shape());
        }
    }
    else if (node_op == "Result")
    {
        shared_ptr<op::Result> res = dynamic_pointer_cast<op::Result>(node);

        if (arg0_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::result(arg0_cipher->get_elements(),
                                        out0_cipher->get_elements(),
                                        shape_size(res->get_shape()));
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
        else
        {
            throw ngraph_error("Slice types not supported.");
        }
    }
    else if (node_op == "Subtract")
    {
        if (arg0_cipher != nullptr && arg1_cipher != nullptr)
        {
            runtime::he::kernel::subtract(arg0_cipher->get_elements(),
                                          arg1_cipher->get_elements(),
                                          out0_cipher->get_elements(),
                                          m_he_backend,
                                          out0_cipher->get_element_count());
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr)
        {
            runtime::he::kernel::subtract(arg0_cipher->get_elements(),
                                          arg1_plain->get_elements(),
                                          out0_cipher->get_elements(),
                                          m_he_backend,
                                          out0_cipher->get_element_count());
        } // TODO: enable (plain, cipher) case
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
            NGRAPH_INFO << "sum plain plain";
            throw ngraph_error("Sum types not supported.");
        }
        else if (arg0_plain != nullptr && out0_cipher != nullptr)
        {
            NGRAPH_INFO << "sum plain -> cipher";
            throw ngraph_error("Sum types not supported.");
        }
        else
        {
            throw ngraph_error("Sum types not supported.");
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
