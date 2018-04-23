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

#include "he_external_function.hpp"
#include "he_backend.hpp"
#include "he_call_frame.hpp"
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/function.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"

using namespace std;
using namespace ngraph;

using descriptor::layout::DenseTensorViewLayout;

runtime::he::HEExternalFunction::HEExternalFunction(const shared_ptr<Function>& function,
                                                    const shared_ptr<HEBackend>& he_backend,
                                                    bool release_function)
    : m_function(function)
    , m_he_backend(he_backend)
    , m_release_function(release_function)
    , m_is_compiled(false)
{
}

void runtime::he::HEExternalFunction::compile()
{
    if (m_is_compiled)
    {
        return;
    }

    pass::Manager pass_manager;
    // For now, just make everyone row-major.
    pass_manager.register_pass<pass::AssignLayout<DenseTensorViewLayout>>();
    pass_manager.register_pass<pass::Liveness>();
    pass_manager.run_passes(m_function);

    m_is_compiled = true;
    if (m_release_function)
    {
        release_function();
    }
}

shared_ptr<runtime::he::HECallFrame> runtime::he::HEExternalFunction::make_call_frame()
{
    if (!m_is_compiled)
    {
        compile();
    }
    return make_shared<runtime::he::HECallFrame>(m_function, m_he_backend);
}
