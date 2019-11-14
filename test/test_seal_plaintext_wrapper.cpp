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

#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_plaintext_wrapper.hpp"
#include "test_util.hpp"

namespace ngraph::he {

TEST(seal_plaintext_wrapper, initialize) {
  // Default plaintext
  {
    auto plain_wrapper = SealPlaintextWrapper(true);
    EXPECT_TRUE(plain_wrapper.complex_packing());

    plain_wrapper.complex_packing() = false;
    EXPECT_FALSE(plain_wrapper.complex_packing());
  }
  // Passed plaintext
  {
    seal::Plaintext plain(100);
    auto plain_wrapper = SealPlaintextWrapper(plain, true);
    EXPECT_EQ(plain, plain_wrapper.plaintext());
  }
}

}  // namespace ngraph::he
