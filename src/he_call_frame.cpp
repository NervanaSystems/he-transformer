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
#include "kernel/constant.hpp"
#include "kernel/dot.hpp"
#include "kernel/multiply.hpp"
#include "kernel/result.hpp"
#include "kernel/subtract.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/dot.hpp"

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
}

void runtime::he::HECallFrame::generate_calls(const element::Type& type,
                                              const shared_ptr<Node>& node,
                                              const vector<shared_ptr<HETensorView>>& args,
                                              const vector<shared_ptr<HETensorView>>& out)
{
    string node_op = node->description();

    if (node_op == "Add")
    {
        shared_ptr<HECipherTensorView> arg0_cipher =
            dynamic_pointer_cast<HECipherTensorView>(args[0]);
        shared_ptr<HECipherTensorView> arg1_cipher =
            dynamic_pointer_cast<HECipherTensorView>(args[1]);
        shared_ptr<HEPlainTensorView> arg0_plain = dynamic_pointer_cast<HEPlainTensorView>(args[0]);
        shared_ptr<HEPlainTensorView> arg1_plain = dynamic_pointer_cast<HEPlainTensorView>(args[1]);
        shared_ptr<HECipherTensorView> out0 = dynamic_pointer_cast<HECipherTensorView>(out[0]);

        if (arg0_cipher != nullptr && arg1_cipher != nullptr)
        {
            runtime::he::kernel::add(arg0_cipher->get_elements(),
                                     arg1_cipher->get_elements(),
                                     out0->get_elements(),
                                     m_he_backend,
                                     out0->get_element_count());
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr)
        {
            runtime::he::kernel::add(arg0_cipher->get_elements(),
                                     arg1_plain->get_elements(),
                                     out0->get_elements(),
                                     m_he_backend,
                                     out0->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_cipher != nullptr)
        {
            runtime::he::kernel::add(arg0_plain->get_elements(),
                                     arg1_cipher->get_elements(),
                                     out0->get_elements(),
                                     m_he_backend,
                                     out0->get_element_count());
        }
        else
        {
            throw ngraph_error("Add types not supported.");
        }
    }
    else if (node_op == "Constant")
    {
        shared_ptr<HEPlainTensorView> out0 = dynamic_pointer_cast<HEPlainTensorView>(out[0]);

        if (out0 != nullptr)
        {
            shared_ptr<op::Constant> constant = static_pointer_cast<op::Constant>(node);
            runtime::he::kernel::constant(out0->get_elements(),
                                          type,
                                          constant->get_data_ptr(),
                                          m_he_backend,
                                          out0->get_element_count());
        }
        else
        {
            throw ngraph_error("Constant type not supported.");
        }
    }
    else if (node_op == "Dot")
    {
        shared_ptr<op::Dot> dot = dynamic_pointer_cast<op::Dot>(node);
        shared_ptr<HECipherTensorView> arg0_cipher =
            dynamic_pointer_cast<HECipherTensorView>(args[0]);
        shared_ptr<HECipherTensorView> arg1_cipher =
            dynamic_pointer_cast<HECipherTensorView>(args[1]);
        shared_ptr<HEPlainTensorView> arg0_plain = dynamic_pointer_cast<HEPlainTensorView>(args[0]);
        shared_ptr<HEPlainTensorView> arg1_plain = dynamic_pointer_cast<HEPlainTensorView>(args[1]);
        shared_ptr<HECipherTensorView> out0 = dynamic_pointer_cast<HECipherTensorView>(out[0]);

        if (arg0_cipher != nullptr && arg1_cipher != nullptr)
        {
            runtime::he::kernel::dot(arg0_cipher->get_elements(),
                                     arg1_cipher->get_elements(),
                                     out0->get_elements(),
                                     arg0_cipher->get_shape(),
                                     arg1_cipher->get_shape(),
                                     out0->get_shape(),
                                     dot->get_reduction_axes_count(),
                                     type,
                                     m_he_backend);
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr)
        {
            runtime::he::kernel::dot(arg0_cipher->get_elements(),
                                     arg1_plain->get_elements(),
                                     out0->get_elements(),
                                     arg0_cipher->get_shape(),
                                     arg1_plain->get_shape(),
                                     out0->get_shape(),
                                     dot->get_reduction_axes_count(),
                                     type,
                                     m_he_backend);
        }
        else if (arg0_plain != nullptr && arg1_cipher != nullptr)
        {
            runtime::he::kernel::dot(arg0_plain->get_elements(),
                                     arg1_cipher->get_elements(),
                                     out0->get_elements(),
                                     arg0_plain->get_shape(),
                                     arg1_cipher->get_shape(),
                                     out0->get_shape(),
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
        shared_ptr<HECipherTensorView> arg0_cipher =
            dynamic_pointer_cast<HECipherTensorView>(args[0]);
        shared_ptr<HECipherTensorView> arg1_cipher =
            dynamic_pointer_cast<HECipherTensorView>(args[1]);
        shared_ptr<HEPlainTensorView> arg0_plain = dynamic_pointer_cast<HEPlainTensorView>(args[0]);
        shared_ptr<HEPlainTensorView> arg1_plain = dynamic_pointer_cast<HEPlainTensorView>(args[1]);
        shared_ptr<HECipherTensorView> out0 = dynamic_pointer_cast<HECipherTensorView>(out[0]);

        if (arg0_cipher != nullptr && arg1_cipher != nullptr)
        {
            runtime::he::kernel::multiply(arg0_cipher->get_elements(),
                                          arg1_cipher->get_elements(),
                                          out0->get_elements(),
                                          m_he_backend,
                                          out0->get_element_count());
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr)
        {
            runtime::he::kernel::multiply(arg0_cipher->get_elements(),
                                          arg1_plain->get_elements(),
                                          out0->get_elements(),
                                          m_he_backend,
                                          out0->get_element_count());
        }
        else if (arg0_plain != nullptr && arg1_cipher != nullptr)
        {
            runtime::he::kernel::multiply(arg0_plain->get_elements(),
                                          arg1_cipher->get_elements(),
                                          out0->get_elements(),
                                          m_he_backend,
                                          out0->get_element_count());
        }
        else
        {
            throw ngraph_error("Multiply types not supported.");
        }
    }
    else if (node_op == "Result")
    {
        shared_ptr<op::Result> res = dynamic_pointer_cast<op::Result>(node);
        shared_ptr<HECipherTensorView> arg0_cipher =
            dynamic_pointer_cast<HECipherTensorView>(args[0]);
        shared_ptr<HEPlainTensorView> arg0_plain = dynamic_pointer_cast<HEPlainTensorView>(args[0]);
        shared_ptr<HECipherTensorView> out0_cipher =
            dynamic_pointer_cast<HECipherTensorView>(out[0]);
        shared_ptr<HEPlainTensorView> out0_plain = dynamic_pointer_cast<HEPlainTensorView>(out[0]);

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
    else if (node_op == "Subtract")
    {
        shared_ptr<HECipherTensorView> arg0_cipher =
            dynamic_pointer_cast<HECipherTensorView>(args[0]);
        shared_ptr<HECipherTensorView> arg1_cipher =
            dynamic_pointer_cast<HECipherTensorView>(args[1]);
        shared_ptr<HEPlainTensorView> arg0_plain = dynamic_pointer_cast<HEPlainTensorView>(args[0]);
        shared_ptr<HEPlainTensorView> arg1_plain = dynamic_pointer_cast<HEPlainTensorView>(args[1]);
        shared_ptr<HECipherTensorView> out0 = dynamic_pointer_cast<HECipherTensorView>(out[0]);

        if (arg0_cipher != nullptr && arg1_cipher != nullptr)
        {
            runtime::he::kernel::subtract(arg0_cipher->get_elements(),
                                          arg1_cipher->get_elements(),
                                          out0->get_elements(),
                                          m_he_backend,
                                          out0->get_element_count());
        }
        else if (arg0_cipher != nullptr && arg1_plain != nullptr)
        {
            runtime::he::kernel::subtract(arg0_cipher->get_elements(),
                                          arg1_plain->get_elements(),
                                          out0->get_elements(),
                                          m_he_backend,
                                          out0->get_element_count());
        } // TODO: enable (plain, cipher) case
        else
        {
            throw ngraph_error("Subtract types not supported.");
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
