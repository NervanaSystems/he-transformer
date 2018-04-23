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

#include "he_call_frame.hpp"
#include "he_backend.hpp"
#include "he_cipher_tensor_view.hpp"
#include "he_tensor_view.hpp"
#include "ngraph/op/result.hpp"

using namespace std;
using namespace ngraph;

runtime::he::HECallFrame::HECallFrame(const shared_ptr<Function>& func)
    : m_function(func)
{
}

void runtime::he::HECallFrame::call(shared_ptr<Function> function,
                                    const vector<shared_ptr<runtime::he::HETensorView>>& output_tvs,
                                    const vector<shared_ptr<runtime::he::HETensorView>>& input_tvs)
{
    // TODO: see interpreter for how this was originally. Need to generalize to PlaintextCipherTensorViews as well
    unordered_map<descriptor::TensorView*, shared_ptr<runtime::he::HECipherTensorView>> tensor_map;
    size_t arg_index = 0;
    for (shared_ptr<op::Parameter> param : function->get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            descriptor::TensorView* tv = param->get_output_tensor_view(i).get();
            shared_ptr<runtime::he::HECipherTensorView> hetv =
                static_pointer_cast<runtime::he::HECipherTensorView>(input_tvs[arg_index++]);
            tensor_map.insert({tv, hetv});
        }
    }

    for (size_t i = 0; i < function->get_output_size(); i++)
    {
        auto output_op = function->get_output_op(i);
        if (!dynamic_pointer_cast<op::Result>(output_op))
        {
            throw ngraph_error("One of function's outputs isn't op::Result");
        }
        descriptor::TensorView* tv = function->get_output_op(i)->get_output_tensor_view(0).get();
        shared_ptr<runtime::he::HECipherTensorView> hetv =
            static_pointer_cast<runtime::he::HECipherTensorView>(output_tvs[i]);
        tensor_map.insert({tv, hetv});
    }

    // Invoke computation
    for (shared_ptr<Node> op : function->get_ordered_ops())
    {
        if (op->description() == "Parameter")
        {
            continue;
        }

        vector<shared_ptr<runtime::he::HETensorView>> inputs;
        vector<shared_ptr<runtime::he::HETensorView>> outputs;
        for (const descriptor::Input& input : op->get_inputs())
        {
            descriptor::TensorView* tv = input.get_output().get_tensor_view().get();
            string name = tv->get_tensor().get_name();
            inputs.push_back(tensor_map.at(tv));
        }
        for (size_t i = 0; i < op->get_output_size(); ++i)
        {
            descriptor::TensorView* tv = op->get_output_tensor_view(i).get();
            string name = tv->get_tensor().get_name();
            shared_ptr<runtime::he::HECipherTensorView> itv;
            if (!contains_key(tensor_map, tv))
            {
                // The output tensor is not in the tensor map so create a new tensor
                const Shape& shape = op->get_output_shape(i);
                const element::Type& element_type = op->get_output_element_type(i);
                string tensor_name = op->get_output_tensor(i).get_name();
                itv = make_shared<runtime::he::HECipherTensorView>(
                    element_type,
                    shape,
                    m_he_backend); // TODO: include tensor name
                tensor_map.insert({tv, itv});
            }
            else
            {
                itv = tensor_map.at(tv);
            }
            outputs.push_back(itv);
        }

        element::Type base_type;
        element::Type secondary_type;
        if (op->get_inputs().empty())
        {
            base_type = op->get_element_type();
        }
        else
        {
            base_type = op->get_inputs().at(0).get_tensor().get_element_type();
        }
        secondary_type = op->get_element_type();

        // Some ops have unusual input/output types so handle those special cases here
        if (op->description() == "Select")
        {
            base_type = op->get_inputs().at(1).get_tensor().get_element_type();
            secondary_type = op->get_inputs().at(0).get_tensor().get_element_type();
        }

        generate_calls(base_type, secondary_type, *op, inputs, outputs);

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

void runtime::he::HECallFrame::generate_calls(
    const element::Type& base_type,
    const element::Type& secondary_type,
    ngraph::Node& op,
    const std::vector<std::shared_ptr<HETensorView>>& args,
    const std::vector<std::shared_ptr<HETensorView>>& out)
{
    if (base_type == element::i64)
    {
        generate_calls<int64_t>(secondary_type, op, args, out);
    }
    else if (base_type == element::u64)
    {
        generate_calls<uint64_t>(secondary_type, op, args, out);
    }
    else if (base_type == element::f64)
    {
        generate_calls<double>(secondary_type, op, args, out);
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
