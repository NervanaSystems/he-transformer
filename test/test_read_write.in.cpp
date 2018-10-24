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
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

#include "he_backend.hpp"
// #include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/bfv/he_seal_bfv_backend.hpp"

#include "test_util.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

TEST(${BACKEND_NAME}, backend_init)
{
    auto he_seal = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    EXPECT_EQ(1, 1);
}
TEST(${BACKEND_NAME}, cipher_tv_write_read_scalar)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{};
    auto a = backend->create_tensor(element::i64, shape);
    copy_he_data(a, vector<int64_t>{5}, backend);
    auto tmp =read_he_vector<int64_t>(a, backend);
    EXPECT_EQ(tmp, (vector<int64_t>{5}));
}

TEST(${BACKEND_NAME}, cipher_tv_write_read_2)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2};
    auto a = backend->create_tensor(element::i64, shape);
    copy_he_data(a, vector<int64_t>{5, 6}, backend);
    EXPECT_EQ(read_he_vector<int64_t>(a, backend), (vector<int64_t>{5, 6}));
}

TEST(${BACKEND_NAME}, cipher_tv_write_read_2_3)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 3};
    auto a = backend->create_tensor(element::i64, shape);
    copy_he_data(a, test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector(), backend);
    EXPECT_EQ(read_he_vector<int64_t>(a, backend),
              (test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector());
}

TEST(${BACKEND_NAME}, plain_tv_write_read_scalar)
{
    auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    Shape shape{};
    auto a = backend->create_plain_tensor(element::i64, shape);
    copy_he_data(a, vector<int64_t>{5}, backend);
    EXPECT_EQ(read_he_vector<int64_t>(a, backend), (vector<int64_t>{5}));
}

TEST(${BACKEND_NAME}, plain_tv_write_read_2)
{
    auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    Shape shape{2};
    auto a = backend->create_plain_tensor(element::i64, shape);
    copy_he_data(a, vector<int64_t>{5, 6}, backend);
    EXPECT_EQ(read_he_vector<int64_t>(a, backend), (vector<int64_t>{5, 6}));
}

TEST(${BACKEND_NAME}, plain_tv_write_read_2_3)
{
    auto backend = dynamic_pointer_cast<runtime::he::HEBackend>(runtime::Backend::create("${BACKEND_REGISTERED_NAME}"));

    Shape shape{2, 3};
    auto a = backend->create_plain_tensor(element::i64, shape);
    copy_he_data(a, test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector(), backend);
    EXPECT_EQ(read_he_vector<int64_t>(a, backend),
              (test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector());
}