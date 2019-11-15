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
#include "test_util.hpp"

namespace ngraph::runtime::he {

TEST(he_plaintext, initialize) {
  {
    std::initializer_list<double> data{1, 2, 3};
    HEPlaintext plain(std::move(data));
    EXPECT_EQ(plain.size(), 3);
    EXPECT_DOUBLE_EQ(plain[0], 1.0);
    EXPECT_DOUBLE_EQ(plain[1], 2.0);
    EXPECT_DOUBLE_EQ(plain[2], 3.0);
  }
  {
    std::vector<double> data{1, 2, 3};
    HEPlaintext plain(std::move(data));
    EXPECT_EQ(plain.size(), 3);
    EXPECT_DOUBLE_EQ(plain[0], 1.0);
    EXPECT_DOUBLE_EQ(plain[1], 2.0);
    EXPECT_DOUBLE_EQ(plain[2], 3.0);
  }
  {
    std::vector<double> data{1, 2, 3};
    HEPlaintext plain(data);
    EXPECT_EQ(plain.size(), 3);
    EXPECT_DOUBLE_EQ(plain[0], 1.0);
    EXPECT_DOUBLE_EQ(plain[1], 2.0);
    EXPECT_DOUBLE_EQ(plain[2], 3.0);
  }
  {
    HEPlaintext plain(3, 1);
    EXPECT_EQ(plain.size(), 3);
    EXPECT_DOUBLE_EQ(plain[0], 1.0);
    EXPECT_DOUBLE_EQ(plain[1], 1.0);
    EXPECT_DOUBLE_EQ(plain[2], 1.0);
  }
}

TEST(he_plaintext, write) {
  HEPlaintext plain{1, 2, 3};
  {
    auto type = element::f32;
    auto src = static_cast<char*>(ngraph_malloc(plain.size() * type.size()));
    plain.write(src, type);
    HEPlaintext dest{reinterpret_cast<float*>(src),
                     reinterpret_cast<float*>(src) + plain.size()};
    EXPECT_TRUE(test::all_close(dest, plain));
    ngraph_free(src);
  }

  {
    auto type = element::f64;
    auto src = static_cast<char*>(ngraph_malloc(plain.size() * type.size()));
    plain.write(src, type);
    HEPlaintext dest{reinterpret_cast<double*>(src),
                     reinterpret_cast<double*>(src) + plain.size()};
    EXPECT_TRUE(test::all_close(dest, plain));
    ngraph_free(src);
  }

  {
    auto type = element::i32;
    auto src = static_cast<char*>(ngraph_malloc(plain.size() * type.size()));
    plain.write(src, type);
    HEPlaintext dest{reinterpret_cast<int32_t*>(src),
                     reinterpret_cast<int32_t*>(src) + plain.size()};
    EXPECT_TRUE(test::all_close(dest, plain));
    ngraph_free(src);
  }

  {
    auto type = element::i64;
    auto src = static_cast<char*>(ngraph_malloc(plain.size() * type.size()));
    plain.write(src, type);
    HEPlaintext dest{reinterpret_cast<int64_t*>(src),
                     reinterpret_cast<int64_t*>(src) + plain.size()};
    EXPECT_TRUE(test::all_close(dest, plain));
    ngraph_free(src);
  }

  // Unsupported types
  auto src = static_cast<char*>(ngraph_malloc(plain.size()));
  EXPECT_ANY_THROW(plain.write(src, element::bf16));
  EXPECT_ANY_THROW(plain.write(src, element::f16));
  EXPECT_ANY_THROW(plain.write(src, element::i8));
  EXPECT_ANY_THROW(plain.write(src, element::i16));
  EXPECT_ANY_THROW(plain.write(src, element::u8));
  EXPECT_ANY_THROW(plain.write(src, element::u16));
  EXPECT_ANY_THROW(plain.write(src, element::u32));
  EXPECT_ANY_THROW(plain.write(src, element::u64));
  EXPECT_ANY_THROW(plain.write(src, element::dynamic));
  EXPECT_ANY_THROW(plain.write(src, element::boolean));
  ngraph_free(src);
}

TEST(he_plaintext, ostream) {
  std::stringstream ss;
  HEPlaintext plain{1, 2, 3};
  EXPECT_NO_THROW(ss << plain);
}

}  // namespace ngraph::runtime::he
