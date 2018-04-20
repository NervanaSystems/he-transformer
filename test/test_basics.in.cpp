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
