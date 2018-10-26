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

#include <limits>

#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/function.hpp"

#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/slice.hpp"

#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"

#include "kernel/add.hpp"
#include "kernel/broadcast.hpp"
#include "kernel/convolution.hpp"
#include "kernel/dot.hpp"
#include "kernel/multiply.hpp"
#include "kernel/negate.hpp"
#include "kernel/subtract.hpp"
#include "kernel/result.hpp"
#include "kernel/reshape.hpp"
#include "kernel/slice.hpp"

#include "he_backend.hpp"
#include "he_cipher_tensor.hpp"
#include "he_plain_tensor.hpp"
#include "he_tensor.hpp"

using namespace ngraph;
using namespace std;

using descriptor::layout::DenseTensorLayout;

shared_ptr<runtime::he::HEPlaintext> runtime::he::HEBackend::create_valued_plaintext(
    float value, const element::Type& element_type) const
{
    const string type_name = element_type.c_type_string();
    shared_ptr<runtime::he::HEPlaintext> plaintext = create_empty_plaintext();

    encode(plaintext, (void*)(&value), element_type, 1);
    return plaintext;
}

shared_ptr<runtime::he::HECiphertext> runtime::he::HEBackend::create_valued_ciphertext(
    float value, const element::Type& element_type, size_t batch_size) const
{
    if (batch_size != 1)
    {
        throw ngraph_error("HESealBFVBackend::create_valued_ciphertext only supports batch size 1");
    }
    const string type_name = element_type.c_type_string();
    shared_ptr<runtime::he::HEPlaintext> plaintext = create_valued_plaintext(value, element_type);
    shared_ptr<runtime::he::HECiphertext> ciphertext = create_empty_ciphertext();

    encrypt(ciphertext, plaintext);
    return ciphertext;
}

shared_ptr<runtime::Tensor> runtime::he::HEBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    throw ngraph_error("HE create_tensor unimplemented");
}

shared_ptr<runtime::Tensor>
    runtime::he::HEBackend::create_tensor(const element::Type& element_type,
                                                       const Shape& shape)
{
    auto rc = make_shared<runtime::he::HECipherTensor>(element_type, shape, create_empty_ciphertext());
    return static_pointer_cast<runtime::Tensor>(rc);
}

shared_ptr<runtime::Tensor>
    runtime::he::HEBackend::create_plain_tensor(const element::Type& element_type,
                                                             const Shape& shape)
{
    auto rc = make_shared<runtime::he::HEPlainTensor>(element_type, shape, create_empty_plaintext());
    return static_pointer_cast<runtime::Tensor>(rc);
}

shared_ptr<runtime::Tensor> runtime::he::HEBackend::create_valued_tensor(
    float value, const element::Type& element_type, const Shape& shape)
{
    auto tensor = static_pointer_cast<HECipherTensor>(create_tensor(element_type, shape));
    vector<shared_ptr<runtime::he::HECiphertext>>& cipher_texts = tensor->get_elements();
#pragma omp parallel for
    for (size_t i = 0; i < cipher_texts.size(); ++i)
    {
        cipher_texts[i] = create_valued_ciphertext(value, element_type);
    }
    return tensor;
}

shared_ptr<runtime::Tensor> runtime::he::HEBackend::create_valued_plain_tensor(
    float value, const element::Type& element_type, const Shape& shape)
{
    auto tensor = static_pointer_cast<HEPlainTensor>(create_plain_tensor(element_type, shape));
    vector<shared_ptr<runtime::he::HEPlaintext>>& plain_texts = tensor->get_elements();
#pragma omp parallel for
    for (size_t i = 0; i < plain_texts.size(); ++i)
    {
        plain_texts[i] = create_valued_plaintext(value, element_type);
    }
    return tensor;
}

bool runtime::he::HEBackend::compile(shared_ptr<Function> function)
{
    FunctionInstance& instance = m_function_map[function];
    if (!instance.m_is_compiled)
    {
        instance.m_is_compiled = true;
        pass::Manager pass_manager;
        pass_manager.register_pass<pass::LikeReplacement>();
        pass_manager.register_pass<pass::AssignLayout<DenseTensorLayout>>();
        pass_manager.register_pass<pass::Liveness>();
        pass_manager.run_passes(function);

        for (const shared_ptr<Node>& node : function->get_ordered_ops())
        {
            instance.m_nodes.emplace_back(node);
        }
    }

    return true;
}

bool runtime::he::HEBackend::call(shared_ptr<Function> function,
                                  const vector<shared_ptr<runtime::Tensor>>& outputs,
                                  const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    vector<shared_ptr<runtime::he::HETensor>> he_inputs;
    vector<shared_ptr<runtime::he::HETensor>> he_outputs;
    for (auto tv : inputs)
    {
        he_inputs.push_back(static_pointer_cast<runtime::he::HETensor>(tv));
    }
    for (auto tv : outputs)
    {
        he_outputs.push_back(static_pointer_cast<runtime::he::HETensor>(tv));
    }
    call(function, he_outputs, he_inputs);
}

void runtime::he::HEBackend::clear_function_instance()
{
    m_function_map.clear();
}

void runtime::he::HEBackend::remove_compiled_function(shared_ptr<Function> function)
{
    m_function_map.erase(function);
}

void runtime::he::HEBackend::enable_performance_data(shared_ptr<Function> function, bool enable)
{
    // Enabled by default
}

vector<runtime::PerformanceCounter>
    runtime::he::HEBackend::get_performance_data(shared_ptr<Function> function) const
{
    vector<runtime::PerformanceCounter> rc;
    const FunctionInstance& instance = m_function_map.at(function);
    for (const pair<const Node*, stopwatch> p : instance.m_timer_map)
    {
        rc.emplace_back(p.first->get_name().c_str(),
                        p.second.get_total_microseconds(),
                        p.second.get_call_count());
    }
    return rc;
}

bool runtime::he::HEBackend::call(shared_ptr<Function> function,
                                    const vector<shared_ptr<runtime::he::HETensor>>& output_tvs,
                                    const vector<shared_ptr<runtime::he::HETensor>>& input_tvs)
{
    // TODO: we clear timer at each run for now
   //  m_timer_map.clear();

    // HEAAN may call with batch != 1, so we disabel validate_call here
    // validate_call(function, outputs, inputs);

    compile(function);
    FunctionInstance& instance = m_function_map[function];

    // map function params -> HETensor
    unordered_map<descriptor::Tensor*, shared_ptr<runtime::he::HETensor>> tensor_map;
    size_t input_count = 0;
    for (auto param : function->get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            descriptor::Tensor* tv = param->get_output_tensor_ptr(i).get();
            tensor_map.insert({tv, input_tvs[input_count++]});
        }
    }

    // map function outputs -> HostTensor
    for (size_t output_count = 0; output_count < function->get_output_size(); ++output_count)
    {
        auto output = function->get_output_op(output_count);
        if (!dynamic_pointer_cast<op::Result>(output))
        {
            throw ngraph_error("One of function's outputs isn't op::Result");
        }
        descriptor::Tensor* tv = output->get_output_tensor_ptr(0).get();
        tensor_map.insert({tv, output_tvs[output_count]});
    }

    // for each ordered op in the graph
    for (shared_ptr<Node> op : function->get_ordered_ops())
    {
        NGRAPH_INFO << "\033[1;32m"
                    << "[ " << op->get_name() << " ]"
                    << "\033[0m";
        if (op->description() == "Parameter")
        {
            continue;
        }

        // get op inputs from map
        vector<shared_ptr<runtime::he::HETensor>> inputs;
        for (const descriptor::Input& input : op->get_inputs())
        {
            descriptor::Tensor* tv = input.get_output().get_tensor_ptr().get();
            inputs.push_back(tensor_map.at(tv));
        }

       // get op outputs from map or create
        bool any_batched = false;
        vector<shared_ptr<runtime::he::HETensor>> outputs;
        for (size_t i = 0; i < op->get_output_size(); ++i)
        {
            descriptor::Tensor* tv = op->get_output_tensor_ptr(i).get();
            auto it = tensor_map.find(tv);
            if (it == tensor_map.end())
            {
                // The output tensor is not in the tensor map so create a new tensor
                const Shape& shape = op->get_output_shape(i);
                const element::Type& element_type = op->get_output_element_type(i);
                string name = op->get_output_tensor(i).get_name();

                bool plain_out = all_of(
                    inputs.begin(), inputs.end(), [](shared_ptr<runtime::he::HETensor> input) {
                        return dynamic_pointer_cast<HEPlainTensor>(input) != nullptr;
                    });
                if (plain_out)
                {
                    auto otv = make_shared<runtime::he::HEPlainTensor>(
                        element_type, shape, create_empty_plaintext(), name);
                    tensor_map.insert({tv, otv});
                }
                else
                {
                    bool batched_out =
                        any_of(inputs.begin(),
                               inputs.end(),
                               [](shared_ptr<runtime::he::HETensor> input) {
                                   if (auto input_cipher_tv =
                                           dynamic_pointer_cast<HECipherTensor>(input))
                                   {
                                       return input_cipher_tv->is_batched();
                                   }
                                   else
                                   {
                                       return false;
                                   }
                               });
                    any_batched |= batched_out;

                    auto otv = make_shared<runtime::he::HECipherTensor>(
                        element_type, shape, create_empty_ciphertext(), batched_out, name);
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

        instance.m_timer_map[op.get()].start();
        generate_calls(base_type, op, outputs, inputs);
        instance.m_timer_map[op.get()].stop();

        const string op_name = op->description();

        // Check result with CPU backend
        /*if (is_cpu_check_enabled(op) && !any_batched)
        {
            check_cpu_calls(function, base_type, op, outputs, inputs, false);
        } */

        // delete any obsolete tensors
        for (const descriptor::Tensor* t : op->liveness_free_list)
        {
            for (auto it = tensor_map.begin(); it != tensor_map.end(); ++it)
            {
                if (it->second->get_name() == t->get_name())
                {
                    tensor_map.erase(it);
                    break;
                }
            }
        }

        // Stop stopwatch and print time
        // TODO: currently timer is cleared at each run
        NGRAPH_INFO << "\033[1;31m" << op->get_name() << " took "
                    << instance.m_timer_map[op.get()].get_seconds() << "s"
                    << "\033[0m";
    }

    return true;
}


void runtime::he::HEBackend::generate_calls(const element::Type& type,
                                              const shared_ptr<Node>& node,
                                              const vector<shared_ptr<HETensor>>& out,
                                              const vector<shared_ptr<HETensor>>& args)
{
    string node_op = node->description();
    shared_ptr<HECipherTensor> arg0_cipher = nullptr;
    shared_ptr<HEPlainTensor> arg0_plain = nullptr;
    shared_ptr<HECipherTensor> arg1_cipher = nullptr;
    shared_ptr<HEPlainTensor> arg1_plain = nullptr;
    shared_ptr<HECipherTensor> out0_cipher = dynamic_pointer_cast<HECipherTensor>(out[0]);
    shared_ptr<HEPlainTensor> out0_plain = dynamic_pointer_cast<HEPlainTensor>(out[0]);

    if (args.size() > 0)
    {
        arg0_cipher = dynamic_pointer_cast<HECipherTensor>(args[0]);
        arg0_plain = dynamic_pointer_cast<HEPlainTensor>(args[0]);
    }
    if (args.size() > 1)
    {
        arg1_cipher = dynamic_pointer_cast<HECipherTensor>(args[1]);
        arg1_plain = dynamic_pointer_cast<HEPlainTensor>(args[1]);
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
                                     this,
                                     out0_cipher->get_element_count());
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::add(arg0_cipher->get_elements(),
                                     arg1_plain->get_elements(),
                                     out0_cipher->get_elements(),
                                     type,
                                     this,
                                     out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::add(arg0_plain->get_elements(),
                                     arg1_cipher->get_elements(),
                                     out0_cipher->get_elements(),
                                     type,
                                     this,
                                     out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_plain != nullptr && out0_plain != nullptr)
        {
            runtime::he::kernel::add(arg0_plain->get_elements(),
                                     arg1_plain->get_elements(),
                                     out0_plain->get_elements(),
                                     type,
                                     this,
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
                                             this);
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
                                             this);
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
                                             this);
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
                                             this);
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
                                     this);
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
                                     this);
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
                                     this);
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
                                     this);
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
                                     this,
                                     out0_cipher->get_element_count());
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::multiply(arg0_cipher->get_elements(),
                                     arg1_plain->get_elements(),
                                     out0_cipher->get_elements(),
                                     type,
                                     this,
                                     out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::multiply(arg0_plain->get_elements(),
                                     arg1_cipher->get_elements(),
                                     out0_cipher->get_elements(),
                                     type,
                                     this,
                                     out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_plain != nullptr && out0_plain != nullptr)
        {
            runtime::he::kernel::multiply(arg0_plain->get_elements(),
                                     arg1_plain->get_elements(),
                                     out0_plain->get_elements(),
                                     type,
                                     this,
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
                                     this,
                                     out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && out0_plain != nullptr)
        {
            runtime::he::kernel::negate(arg0_plain->get_elements(),
                                     out0_plain->get_elements(),
                                     type,
                                     this,
                                     out0_plain->get_element_count());
        }
        else
        {
            throw ngraph_error("Negative types not supported.");
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
                                        this);
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
                                     this,
                                     out0_cipher->get_element_count());
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::subtract(arg0_cipher->get_elements(),
                                     arg1_plain->get_elements(),
                                     out0_cipher->get_elements(),
                                     type,
                                     this,
                                     out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_cipher != nullptr && out0_cipher != nullptr)
        {
            runtime::he::kernel::subtract(arg0_plain->get_elements(),
                                     arg1_cipher->get_elements(),
                                     out0_cipher->get_elements(),
                                     type,
                                     this,
                                     out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_plain != nullptr && out0_plain != nullptr)
        {
            runtime::he::kernel::subtract(arg0_plain->get_elements(),
                                     arg1_plain->get_elements(),
                                     out0_plain->get_elements(),
                                     type,
                                     this,
                                     out0_plain->get_element_count());
        }
        else
        {
            throw ngraph_error("Subtract types not supported.");
        }
    }
    else
    {
        NGRAPH_INFO << "Op type " << node_op << " unsupported";
        throw ngraph_error("Unsupported op type");
    }

}