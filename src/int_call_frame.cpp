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

#include "ngraph/runtime/interpreter/int_backend.hpp"
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/util/binary_elementwise_comparison.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

using descriptor::layout::DenseTensorViewLayout;

runtime::interpreter::INT_CallFrame::INT_CallFrame(shared_ptr<Function> func)
    : m_function(func)
{
}

void runtime::interpreter::INT_CallFrame::generate_calls(
    const element::Type& type,
    Node& op,
    const vector<shared_ptr<HostTensorView>>& outputs,
    const vector<shared_ptr<HostTensorView>>& inputs)
{
    if (type == element::boolean)
    {
        op_engine<char>(op, outputs, inputs);
    }
    else if (type == element::f32)
    {
        op_engine<float>(op, outputs, inputs);
    }
    else if (type == element::f64)
    {
        op_engine<double>(op, outputs, inputs);
    }
    else if (type == element::i8)
    {
        op_engine<int8_t>(op, outputs, inputs);
    }
    else if (type == element::i16)
    {
        op_engine<int16_t>(op, outputs, inputs);
    }
    else if (type == element::i32)
    {
        op_engine<int32_t>(op, outputs, inputs);
    }
    else if (type == element::i64)
    {
        op_engine<int64_t>(op, outputs, inputs);
    }
    else if (type == element::u8)
    {
        op_engine<uint8_t>(op, outputs, inputs);
    }
    else if (type == element::u16)
    {
        op_engine<uint16_t>(op, outputs, inputs);
    }
    else if (type == element::u32)
    {
        op_engine<uint32_t>(op, outputs, inputs);
    }
    else if (type == element::u64)
    {
        op_engine<uint64_t>(op, outputs, inputs);
    }
    else
    {
        stringstream ss;
        ss << "unsupported element type " << type << " op " << op.get_name();
        throw ngraph_error(ss.str());
    }
}
