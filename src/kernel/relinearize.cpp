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

#include <vector>

#include "he_backend.hpp"
#include "he_seal_backend.hpp"
#include "kernel/relinearize.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::relinearize(const vector<shared_ptr<runtime::he::HECiphertext>>& arg,
                                      vector<shared_ptr<runtime::he::HECiphertext>>& out,
                                      shared_ptr<runtime::he::he_seal::HESealBackend> he_seal_backend,
                                      size_t count)
{
    throw ngraph_error("Relinearize not implemnted");
// shared_ptr<seal::EvaluationKeys> ev_key = he_seal_backend->get_ev_key();

// It's safe to do inplace relinearize on the input since the un-relinearized result won't be
// used by other ops. That is, this relinearize op is immediately after a multiply op, and the
// relinearize op is the only op using the result from the multiply op
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        // he_seal_backend->get_evaluator()->relinearize(*arg[i], *(he_seal_backend->get_ev_key())); # TODO: enable
        out[i] = arg[i];
    }
}
