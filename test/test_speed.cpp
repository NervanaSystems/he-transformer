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

#include <assert.h>

#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

#include "he_backend.hpp"
#include "test_util.hpp"

using namespace std;
using namespace ngraph;

TEST_F(TestHEBackend, dot_20)
{
    // Shape and raw data
    Shape shape{20, 20};
    vector<float> x_data(400, 1.1);
    vector<float> w_data(400, 1.2);
    vector<float> r_data(400, 26.4);

    // Graph
    auto X = make_shared<op::Parameter>(element::f32, shape);
    auto W = op::Constant::create(element::f32, shape, w_data);
    auto R = make_shared<op::Dot>(X, W);
    auto f = make_shared<Function>(R, op::ParameterVector{X});

    // TensorViews
    auto x = m_he_backend->create_tensor(element::f32, shape);
    auto r = m_he_backend->create_tensor(element::f32, shape);
    copy_data(x, x_data);

    // Compute
    m_he_backend->call(f, {r}, {x});
    EXPECT_TRUE(test::all_close(r_data, read_vector<float>(r)));
}
