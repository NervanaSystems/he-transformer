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

#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/util.hpp"
#include "ngraph/type/element_type.hpp"
#include "he_tensor_view.hpp"
#include "he_cipher_tensor_view.hpp"

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

                /// @brief Invoke the function with values matching the signature of the function.
                ///
                /// Tuples will be expanded into their tensor views to build the call frame.
                void call(const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
                          const std::vector<std::shared_ptr<runtime::TensorView>>& inputs);

                void set_nan_check(bool);

            private:
                std::shared_ptr<Function> m_function;
                HEBackend* he_backend;
                bool m_emit_timing;
                bool m_nan_check;
                std::unordered_map<const Node*, stopwatch> m_timer_map;

                static void perform_nan_check(const std::vector<std::shared_ptr<he::HETensorView>>&,
                        const Node* op = nullptr);

                void call(std::shared_ptr<Function> function,
                          const std::vector<std::shared_ptr<runtime::he::HETensorView>>& output_tvs,
                          const std::vector<std::shared_ptr<runtime::he::HETensorView>>& input_tvs);

                void generate_calls(const element::Type& base_type,
                                    const element::Type& secondary_type,
                                    ngraph::Node& op,
                                    const std::vector<std::shared_ptr<he::HETensorView>>& args,
                                    const std::vector<std::shared_ptr<he::HETensorView>>& out);

                template <typename BASE>
                void generate_calls(const element::Type& type,
                                    ngraph::Node& op,
                                    const std::vector<std::shared_ptr<he::HETensorView>>& args,
                                    const std::vector<std::shared_ptr<he::HETensorView>>& out)
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
					else
					{
						std::stringstream ss;
						ss << "unsupported element type " << type << " op " << op.get_name();
						throw std::runtime_error(ss.str());
					}
                }

				template <typename T, typename S>
				void op_engine(ngraph::Node& node,
							   const std::vector<std::shared_ptr<he::HETensorView>>& args,
							   const std::vector<std::shared_ptr<he::HETensorView>>& out)
				{
					std::string node_op = node.description();
                    std::stringstream ss;
                    ss << "unsupported op " << node_op;
                    throw std::runtime_error(ss.str());
                }
            };
        }
    }
}
