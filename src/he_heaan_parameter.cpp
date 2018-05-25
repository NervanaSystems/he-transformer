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

#include "he_heaan_parameter.hpp"

using namespace ngraph;
using namespace std;

runtime::he::HEHeaanParameter::HEHeaanParameter(uint64_t log_poly_modulus,
                                                uint64_t log_plain_modulus,
                                                uint64_t log_precision)
    : HEParameter(log_poly_modulus, log_plain_modulus)
    , m_log_precision(log_precision)
{
}
