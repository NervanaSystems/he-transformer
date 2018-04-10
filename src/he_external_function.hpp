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

#include <memory>

#include "ngraph/runtime/external_function.hpp"

namespace ngraph
{
    class Function;

    namespace runtime
    {
        class CallFrame;

        namespace he
        {
            class HEExternalFunction : public runtime::ExternalFunction
            {
            public:
                HEExternalFunction(const std::shared_ptr<Function>& function,
                                   bool release_function = false);
                std::shared_ptr<runtime::CallFrame> make_call_frame();

            protected:
                void compile();
            };
        }
    }
}
