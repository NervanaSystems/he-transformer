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

#include "ngraph/op/result.hpp"

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
#include "kernel/reshape.hpp"
#include "kernel/result.hpp"
#include "kernel/slice.hpp"
#include "kernel/subtract.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"

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
        if (op->description() == "Parameter")
        {
            continue;
        }

        // Collect input runtime::tv
        vector<shared_ptr<runtime::he::HETensorView>> inputs;
        for (const descriptor::Input& input : op->get_inputs())
        {
            descriptor::TensorView* tv = input.get_output().get_tensor_view().get();
            string name = tv->get_tensor().get_name();
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
                if (op->description() == "Constant")
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

        NGRAPH_INFO << "Op " << op->get_name();
        generate_calls(base_type, op, inputs, outputs);

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
    }
    // Check noise budget
    NGRAPH_INFO << "Checking noise budget ";
#pragma omp parallel for
    for (size_t i = 0; i < output_tvs.size(); ++i)
    {
        shared_ptr<HECipherTensorView> out_i =
            dynamic_pointer_cast<HECipherTensorView>(output_tvs[i]);
        if (out_i != nullptr)
        {
            for (shared_ptr<seal::Ciphertext> ciphertext : out_i->get_elements())
            {
                if (m_he_backend->noise_budget(ciphertext) <= 0)
                {
                    throw ngraph_error("Noise budget depleted");
                }
            }
        }
    }
    NGRAPH_INFO << "Done checking noise budget ";
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
        else
        {
            throw ngraph_error("Add types not supported.");
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
                                          m_he_backend,
                                          out0_cipher->get_element_count());
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr)
        {
            runtime::he::kernel::multiply(arg0_cipher->get_elements(),
                                          arg1_plain->get_elements(),
                                          out0_cipher->get_elements(),
                                          m_he_backend,
                                          out0_cipher->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_cipher != nullptr)
        {
            runtime::he::kernel::multiply(arg0_plain->get_elements(),
                                          arg1_cipher->get_elements(),
                                          out0_cipher->get_elements(),
                                          m_he_backend,
                                          out0_cipher->get_element_count());
        }
        else
        {
            throw ngraph_error("Multiply types not supported.");
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
        else if (arg0_plain != nullptr && out0_cipher != nullptr)
        {
            Shape in_shape = arg0_plain->get_shape();
            Shape out_shape = out0_cipher->get_shape();
            runtime::he::kernel::broadcast(arg0_plain->get_elements(),
                                           out0_cipher->get_elements(),
                                           in_shape,
                                           out_shape,
                                           broadcast_axes,
                                           m_he_backend);
        }
        // TODO: enable (plain, cipher) and (plain, plain) cases
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
