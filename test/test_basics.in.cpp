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

#include "gtest/gtest.h"
#include "ngraph/log.hpp"

#include "util/test_tools.hpp"

TEST(he_transformer, trivial)
{
    int a = 1;
    int b = 2;
    EXPECT_EQ(3, a + b);
}

TEST(he_transformer, backend_init)
{
    auto backend = runtime::Backend::create("HE");
    EXPECT_EQ(1, 1);
}

TEST(he_transformer, tensor_read_write)
{
    auto backend = runtime::Backend::create("HE");
    Shape shape{2, 3};
    shared_ptr<runtime::TensorView> a = backend->create_tensor<int64_t>(shape);
    copy_data(a, test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_EQ(read_vector<int64_t>(a),
              (test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector());
}

TEST(he_transformer, ab)
{
	Shape s{2, 3};
	auto a = std::make_shared<op::Parameter>(element::f64, s);
	auto b = std::make_shared<op::Parameter>(element::f64, s);

	auto t0 = std::make_shared<op::Add>(a, b);

	// Make the function
	auto f = std::make_shared<Function>(NodeVector{t0},
			op::ParameterVector{a, b});

	// Create the backend
	auto backend = runtime::Backend::create("HE");

	// Allocate tensors for arguments a, b, c
	auto t_a = backend->create_tensor(element::f64, s);
	auto t_b = backend->create_tensor(element::f64, s);
	// Allocate tensor for the result
	auto t_result = backend->create_tensor(element::f64, s);

	// Initialize tensors
	double v_a[2][3] = {{1, 2, 3}, {4, 5, 6}};
	double v_b[2][3] = {{7, 8, 9}, {10, 11, 12}};

	t_a->write(&v_a, 0, sizeof(v_a));
	t_b->write(&v_b, 0, sizeof(v_b));
	std::cout << "wrote " << std::endl;

	// Invoke the function
	backend->call(f, {t_result}, {t_a, t_b});

	// Get the result
	double r[2][3];
	t_result->read(&r, 0, sizeof(r));

    std::cout << "[" << std::endl;
    for (size_t i = 0; i < s[0]; ++i)
    {
        std::cout << " [";
        for (size_t j = 0; j < s[1]; ++j)
        {
            std::cout << r[i][j] << ' ';
        }
        std::cout << ']' << std::endl;
    }
    std::cout << ']' << std::endl;


}
