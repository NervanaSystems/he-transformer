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

TEST(he_plaintext, initialize) {
  {
    std::initializer_list<double> data{1, 2, 3};
    ngraph::he::HEPlaintext plain(std::move(data));
    EXPECT_EQ(plain.size(), 3);
    EXPECT_DOUBLE_EQ(plain[0], 1.0);
    EXPECT_DOUBLE_EQ(plain[1], 2.0);
    EXPECT_DOUBLE_EQ(plain[2], 3.0);
  }
  {
    std::vector<double> data{1, 2, 3};
    ngraph::he::HEPlaintext plain(std::move(data));
    EXPECT_EQ(plain.size(), 3);
    EXPECT_DOUBLE_EQ(plain[0], 1.0);
    EXPECT_DOUBLE_EQ(plain[1], 2.0);
    EXPECT_DOUBLE_EQ(plain[2], 3.0);
  }
  {
    std::vector<double> data{1, 2, 3};
    ngraph::he::HEPlaintext plain(data);
    EXPECT_EQ(plain.size(), 3);
    EXPECT_DOUBLE_EQ(plain[0], 1.0);
    EXPECT_DOUBLE_EQ(plain[1], 2.0);
    EXPECT_DOUBLE_EQ(plain[2], 3.0);
  }
  {
    ngraph::he::HEPlaintext plain(3, 1);
    EXPECT_EQ(plain.size(), 3);
    EXPECT_DOUBLE_EQ(plain[0], 1.0);
    EXPECT_DOUBLE_EQ(plain[1], 1.0);
    EXPECT_DOUBLE_EQ(plain[2], 1.0);
  }
}

TEST(he_plaintext, write) {
  ngraph::he::HEPlaintext plain{1, 2, 3};
  {
    auto type = ngraph::element::f32;
    auto src =
        static_cast<char*>(ngraph::ngraph_malloc(plain.size() * type.size()));
    plain.write(src, type);
    ngraph::he::HEPlaintext dest{reinterpret_cast<float*>(src),
                                 reinterpret_cast<float*>(src) + plain.size()};
    EXPECT_TRUE(ngraph::test::he::all_close(dest, plain));
    ngraph::ngraph_free(src);
  }

  {
    auto type = ngraph::element::f64;
    auto src =
        static_cast<char*>(ngraph::ngraph_malloc(plain.size() * type.size()));
    plain.write(src, type);
    ngraph::he::HEPlaintext dest{reinterpret_cast<double*>(src),
                                 reinterpret_cast<double*>(src) + plain.size()};
    EXPECT_TRUE(ngraph::test::he::all_close(dest, plain));
    ngraph::ngraph_free(src);
  }

  {
    auto type = ngraph::element::i32;
    auto src =
        static_cast<char*>(ngraph::ngraph_malloc(plain.size() * type.size()));
    plain.write(src, type);
    ngraph::he::HEPlaintext dest{
        reinterpret_cast<int32_t*>(src),
        reinterpret_cast<int32_t*>(src) + plain.size()};
    EXPECT_TRUE(ngraph::test::he::all_close(dest, plain));
    ngraph::ngraph_free(src);
  }

  {
    auto type = ngraph::element::i64;
    auto src =
        static_cast<char*>(ngraph::ngraph_malloc(plain.size() * type.size()));
    plain.write(src, type);
    ngraph::he::HEPlaintext dest{
        reinterpret_cast<int64_t*>(src),
        reinterpret_cast<int64_t*>(src) + plain.size()};
    EXPECT_TRUE(ngraph::test::he::all_close(dest, plain));
    ngraph::ngraph_free(src);
  }

  // Unsupported types
  auto src = static_cast<char*>(ngraph::ngraph_malloc(plain.size()));
  EXPECT_ANY_THROW(plain.write(src, ngraph::element::bf16));
  EXPECT_ANY_THROW(plain.write(src, ngraph::element::f16));
  EXPECT_ANY_THROW(plain.write(src, ngraph::element::i8));
  EXPECT_ANY_THROW(plain.write(src, ngraph::element::i16));
  EXPECT_ANY_THROW(plain.write(src, ngraph::element::u8));
  EXPECT_ANY_THROW(plain.write(src, ngraph::element::u16));
  EXPECT_ANY_THROW(plain.write(src, ngraph::element::u32));
  EXPECT_ANY_THROW(plain.write(src, ngraph::element::u64));
  EXPECT_ANY_THROW(plain.write(src, ngraph::element::dynamic));
  EXPECT_ANY_THROW(plain.write(src, ngraph::element::boolean));
  ngraph::ngraph_free(src);
}

TEST(he_plaintext, ostream) {
  std::stringstream ss;
  ngraph::he::HEPlaintext plain{1, 2, 3};
  EXPECT_NO_THROW(ss << plain);
}
