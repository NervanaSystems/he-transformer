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

#include <functional>
#include <memory>
#include <vector>

#include "he_cipher_tensor_view.hpp"
#include "he_tensor_view.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"
#include "op/add.hpp"
#include "op/result.hpp"

namespace ngraph
{
    class Function;

    namespace runtime
    {
        class TensorView;
        class ExternalFunction;

        namespace he
        {
            class HEExternalFunction;
            class HETensorView;

            // A VM for executing lightly-compiled graph functions
            class HECallFrame
            {
                friend class HEBackend;

            public:
                HECallFrame(const std::shared_ptr<Function>& function);

                void call(const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
                          const std::vector<std::shared_ptr<runtime::TensorView>>& inputs);

            private:
                std::shared_ptr<Function> m_function;
                std::shared_ptr<HEBackend> m_he_backend;

                void call(std::shared_ptr<Function> function,
                          const std::vector<std::shared_ptr<runtime::he::HETensorView>>& output_tvs,
                          const std::vector<std::shared_ptr<runtime::he::HETensorView>>& input_tvs);

				void generate_calls(const element::Type& base_type,
						const element::Type& secondary_type,
						ngraph::Node& op,
						const std::vector<std::shared_ptr<HETensorView>>& args,
						const std::vector<std::shared_ptr<HETensorView>>& out);

				template <typename BASE>
					void generate_calls(const element::Type& type,
							ngraph::Node& op,
							const std::vector<std::shared_ptr<HETensorView>>& args,
							const std::vector<std::shared_ptr<HETensorView>>& out)
					{
						if (type == element::f64)
						{
							op_engine<BASE, double>(op, args, out);
						}
						else if (type == element::i64)
						{
							op_engine<BASE, int64_t>(op, args, out);
						}
						else if (type == element::u64)
						{
							op_engine<BASE, uint64_t>(op, args, out);
						}
					}

				template <typename T, typename S>
					void op_engine(ngraph::Node& node,
							const std::vector<std::shared_ptr<HETensorView>>& args,
							const std::vector<std::shared_ptr<HETensorView>>& out)
					{
						std::string node_op = node.description();

                        if (node_op == "Add")
                        {
                            HECipherTensorView* arg0 = dynamic_cast<HECipherTensorView*>(args[0].get());
                            HECipherTensorView* arg1 = dynamic_cast<HECipherTensorView*>(args[1].get());
                            HECipherTensorView* out0 = dynamic_cast<HECipherTensorView*>(out[0].get());

                            runtime::he::add(arg0, arg1, out0, out0->get_element_count());
                        }
                        else if(node_op == "Result")
                        {
                            ngraph::op::Result* res = dynamic_cast<ngraph::op::Result*>(&node);
                            runtime::he::result(dynamic_cast<HECipherTensorView*>(args[0].get()),
                                    dynamic_cast<HECipherTensorView*>(out[0].get()),
                                    shape_size(res->get_shape()));
                        }
                        else
                        {
                            throw ngraph_error("node op " + node_op + " unimplemented");
                        }
					}
            };
        }
    }
}
