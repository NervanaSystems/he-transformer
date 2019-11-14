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

#include "ngraph/ngraph.hpp"
#include "seal/he_seal_backend.hpp"
#include "test_util.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

namespace ngraph::runtime::he {

TEST(he_unsupported_ops, op) {
  auto backend = runtime::Backend::create("HE_SEAL");

  Shape shape{11};
  auto a = std::make_shared<op::Parameter>(element::f32, shape);
  auto f = std::make_shared<Function>(std::make_shared<op::Cos>(a),
                                      ParameterVector{a});

  EXPECT_THROW({ backend->compile(f); }, CheckFailure);
}

TEST(he_unsupported_ops, element_type) {
  auto backend = runtime::Backend::create("HE_SEAL");

  Shape shape{11};
  auto a = std::make_shared<op::Parameter>(element::i8, shape);
  auto b = std::make_shared<op::Parameter>(element::i8, shape);
  {
    auto f = std::make_shared<Function>(std::make_shared<op::Add>(a, b),
                                        ParameterVector{a, b});
    EXPECT_THROW({ backend->compile(f); }, CheckFailure);
  }
  {
    auto f = std::make_shared<Function>(std::make_shared<op::Multiply>(a, b),
                                        ParameterVector{a, b});
    EXPECT_THROW({ backend->compile(f); }, CheckFailure);
  }
  {
    auto f = std::make_shared<Function>(std::make_shared<op::Subtract>(a, b),
                                        ParameterVector{a, b});
    EXPECT_THROW({ backend->compile(f); }, CheckFailure);
  }
}

}  // namespace ngraph::runtime::he
