//*****************************************************************************
// Copyright 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <assert.h>

#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

#include "he_backend.hpp"
#include "test_util.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, backend_init)
{
    NGRAPH_INFO << "Initializing backend ${BACKEND_REGISTERED_NAME}";
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    EXPECT_EQ(1, 1);
}

NGRAPH_TEST(${BACKEND_NAME}, cipher_tv_write_read_scalar)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{5});
    EXPECT_TRUE(all_close(read_vector<float>(a), (vector<float>{5})));
}

NGRAPH_TEST(${BACKEND_NAME}, cipher_tv_write_read_2)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{5, 6});
    EXPECT_TRUE(all_close(read_vector<float>(a), (vector<float>{5, 6})));
}

NGRAPH_TEST(${BACKEND_NAME}, cipher_tv_write_read_2_3)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 3};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_TRUE(all_close(read_vector<float>(a),
                          (test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector()));
}

NGRAPH_TEST(${BACKEND_NAME}, cipher_tv_write_read_5_5)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{5, 5};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, test::NDArray<float, 2>({{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}}).get_vector());
    EXPECT_TRUE(all_close(read_vector<float>(a),
                          (test::NDArray<float, 2>({{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}})).get_vector()));
}

NGRAPH_TEST(${BACKEND_NAME}, plain_tv_write_read_scalar)
{
    auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    Shape shape{};
    auto a = backend->create_plain_tensor(element::f32, shape);
    copy_data(a, vector<float>{5});
    EXPECT_TRUE(all_close(read_vector<float>(a), (vector<float>{5})));
}

NGRAPH_TEST(${BACKEND_NAME}, plain_tv_write_read_2)
{
    auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    Shape shape{2};
    auto a = backend->create_plain_tensor(element::f32, shape);
    copy_data(a, vector<float>{5, 6});
    EXPECT_TRUE(all_close(read_vector<float>(a), (vector<float>{5, 6})));
}

NGRAPH_TEST(${BACKEND_NAME}, plain_tv_write_read_2_3)
{
    auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    Shape shape{2, 3};
    auto a = backend->create_plain_tensor(element::f32, shape);
    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_TRUE(all_close(read_vector<float>(a),
                          test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector()));
}

NGRAPH_TEST(${BACKEND_NAME}, cipher_tv_batch)
{
    auto backend = static_pointer_cast<runtime::he::HEBackend>(
        runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    Shape shape{2, 3};
    auto a = backend->create_batched_tensor(element::f32, shape);
    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());

    EXPECT_EQ(generalized_read_vector<float>(a),
              (test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector());
}
