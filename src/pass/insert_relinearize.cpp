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

#include "ngraph/graph_util.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/multiply.hpp"

#include "op/relinearize.hpp"
#include "pass/insert_relinearize.hpp"

using namespace std;
using namespace ngraph;

bool runtime::he::pass::InsertRelinearize::run_on_call_graph(const list<shared_ptr<Node>>& nodes)
{
    for (const shared_ptr<Node>& node : nodes)
    {
        if (auto multiply = dynamic_pointer_cast<op::Multiply>(node))
        {
            shared_ptr<Node> new_node =
                node->copy_with_new_args(NodeVector{node->get_argument(0), node->get_argument(1)});
            new_node = make_shared<op::Relinearize>(new_node);
            replace_node(multiply, new_node);
        }
        else if (auto multiply = dynamic_pointer_cast<op::Dot>(node))
        {
            shared_ptr<Node> new_node =
                node->copy_with_new_args(NodeVector{node->get_argument(0), node->get_argument(1)});
            new_node = make_shared<op::Relinearize>(new_node);
            replace_node(multiply, new_node);
        }
    }
    return false;
}
