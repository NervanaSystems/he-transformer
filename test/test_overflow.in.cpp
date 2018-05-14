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

#include "seal_parameter.hpp"

TEST_F(TestHEBackend, overflow)
{
    // Shape and raw data
    Shape shape{2, 2};
    vector<float> x_data(4, 1.1);
    vector<float> w_data(4, 1.2);
    vector<float> r_data(4, 26.4);

    // Graph
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = op::Constant::create(element::f32, shape, w_data);
    auto C = make_shared<op::Dot>(A, B);
    auto D = make_shared<op::Dot>(A, C);
    auto E = make_shared<op::Dot>(A, D);
    auto F = make_shared<op::Dot>(A, E);
    auto G = make_shared<op::Dot>(A, F);
    auto H = make_shared<op::Dot>(A, G);
    auto I = make_shared<op::Dot>(A, H);
    auto f = make_shared<Function>(G, op::ParameterVector{A});

    // TensorViews
    auto x = m_he_backend->create_tensor(element::f32, shape);
    auto r = m_he_backend->create_tensor(element::f32, shape);
    copy_data(x, x_data);

    // Compute
    m_he_backend->call(f, {r}, {x});
    auto res = read_vector<float>(r);

    cout << "result" << endl;
    for (auto elem : res)
    {
        cout << elem << " " ;
    }
    cout << endl;
    //EXPECT_TRUE(test::all_close(r_data, read_vector<float>(r)));
}
