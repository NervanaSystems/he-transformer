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

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/requires_tensor_view_args.hpp"
#include "ngraph/type/type.hpp"

namespace ngraph
{
    namespace op
    {
        class Square : public ngraph::op::util::RequiresTensorViewArgs
        {
        public:
            Square(const std::shared_ptr<Node>& arg)
                : RequiresTensorViewArgs("Square", {arg})
            {
                set_value_type_checked(get_inputs().at(0).get_element_type(), arg->get_shape());
            }
            std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const
            {
                throw ngraph_error("Cannot copy this node");
            }
        };
    }
}
