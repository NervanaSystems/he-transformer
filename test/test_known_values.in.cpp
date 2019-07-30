//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
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

#include <memory>
#include <vector>

#include "ngraph/ngraph.hpp"
#include "seal/he_seal_backend.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

#include "seal/kernel/add_seal.hpp"
#include "seal/kernel/multiply_seal.hpp"
#include "seal/kernel/negate_seal.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, known_value_cipher_cipher_add) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  size_t count = 10;

  vector<shared_ptr<ngraph::he::SealCiphertextWrapper>> arg1;
  vector<shared_ptr<ngraph::he::SealCiphertextWrapper>> arg2;
  vector<shared_ptr<ngraph::he::SealCiphertextWrapper>> out;
  vector<float> exp_out(count);

  for (size_t i = 0; i < count; i++) {
    arg1.emplace_back(make_shared<ngraph::he::SealCiphertextWrapper>());
    arg2.emplace_back(make_shared<ngraph::he::SealCiphertextWrapper>());
    out.emplace_back(make_shared<ngraph::he::SealCiphertextWrapper>());

    arg1[i]->known_value() = true;
    arg1[i]->value() = i;

    arg2[i]->known_value() = true;
    arg2[i]->value() = i * 1.23;

    exp_out[i] = arg1[i]->value() + arg2[i]->value();
  }
  ngraph::he::add_seal(arg1, arg2, out, ngraph::element::f32, *he_backend,
                       count);

  for (size_t i = 0; i < count; ++i) {
    auto c_out = out[i];
    EXPECT_TRUE(c_out->known_value());
    EXPECT_EQ(c_out->value(), exp_out[i]);
  }
}