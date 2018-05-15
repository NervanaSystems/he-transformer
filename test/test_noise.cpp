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

#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

#include "he_backend.hpp"
#include "seal_parameter.hpp"
#include "test_util.hpp"

using namespace std;
using namespace ngraph;

// TODO
// - [ ] Max depth of Mul
// - [ ] Max depth of Add
// - [ ] A * B * C * D vs (A * B) * (C * D)
// - [ ] FractionalEncoder vs IntergerEncoder, noise
// - [ ] Refactor noise check

// struct SEALParameter
// {
//     std::uint64_t poly_modulus_degree;
//     std::uint64_t security_level;
//     std::uint64_t plain_modulus;
//     int fractional_encoder_integer_coeff_count;
//     int fractional_encoder_fraction_coeff_count;
//     std::uint64_t fractional_encoder_base;
//     int evaluation_decomposition_bit_count;
// };

TEST_F(TestHEBackend, noise)
{
    Shape shape{};
    runtime::he::SEALParameter seal_parameter{16384, 128, 50000, 64, 32, 3, 16};
    auto he_backend = make_shared<runtime::he::HEBackend>();
    auto a = he_backend->create_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{5});
    EXPECT_EQ(read_vector<int64_t>(a), (vector<int64_t>{5}));
}
