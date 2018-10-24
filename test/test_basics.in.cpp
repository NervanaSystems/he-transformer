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

TEST(${BACKEND_NAME}, trivial)
{
    int a = 1;
    int b = 2;
    EXPECT_EQ(3, a + b);
}

TEST(backend_api, registered_devices)
{
    vector<string> devices = runtime::Backend::get_registered_devices();
    for (auto elem : devices)
    {
        NGRAPH_INFO << "device " << elem;
    }
}

TEST(${BACKEND_NAME}, backend_init)
{
    auto he_seal = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    NGRAPH_INFO << "Created SEAL BFV backend";
    EXPECT_EQ(1, 1);
}
TEST(${BACKEND_NAME}, cipher_tv_write_read_scalar)
{
    NGRAPH_INFO << "Creating backend";
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");
    NGRAPH_INFO << "Created backend";

    Shape shape{};
    auto a = backend->create_tensor(element::i64, shape);
    NGRAPH_INFO << "Created tensor";
    copy_he_data(a, vector<int64_t>{5}, backend);
    NGRAPH_INFO << "Copied he data";
    auto tmp =read_he_vector<int64_t>(a, backend);
    NGRAPH_INFO << "Read he vector";
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

TEST(${BACKEND_NAME}, ab)
{
    auto backend = runtime::Backend::create("${BACKEND_REGISTERED_NAME}");

    Shape shape{2, 3};
    auto a = make_shared<op::Parameter>(element::i64, shape);
    auto b = make_shared<op::Parameter>(element::i64, shape);
    auto t = make_shared<op::Add>(a, b);
    auto f = make_shared<Function>(t, op::ParameterVector{a, b});

    // Create some tensors for input/output
    auto tensors_list = generate_plain_cipher_tensors({t}, {a, b}, backend);

    for (auto tensors : tensors_list)
    {
        auto results = get<0>(tensors);
        auto inputs = get<1>(tensors);

        auto t_a = inputs[0];
        auto t_b = inputs[1];
        auto t_result = results[0];

        copy_he_data(t_a, test::NDArray<int64_t, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector(), backend);
        copy_he_data(t_b, test::NDArray<int64_t, 2>({{7, 8, 9}, {10, 11, 12}}).get_vector(), backend);
        backend->call(f, {t_result}, {t_a, t_b});
        EXPECT_EQ(read_he_vector<int64_t>(t_result, backend),
                  (test::NDArray<int64_t, 2>({{8, 10, 12}, {14, 16, 18}})).get_vector());
    }
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