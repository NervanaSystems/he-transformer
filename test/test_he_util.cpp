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

#include <complex>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "he_util.hpp"
#include "ngraph/type/element_type.hpp"
#include "test_util.hpp"
#include "util/test_tools.hpp"

TEST(he_util, complex_vec_to_real_vec) {
  {
    std::vector<std::complex<double>> complex_vec{{1, 2}, {3, 4}};
    std::vector<double> real_vec;
    ngraph::he::complex_vec_to_real_vec(real_vec, complex_vec);

    EXPECT_EQ(real_vec.size(), 2 * complex_vec.size());
    EXPECT_TRUE(
        ngraph::test::he::all_close(real_vec, std::vector<double>{1, 2, 3, 4}));
  }

  {
    std::vector<std::complex<double>> complex_vec{{-2, -1}, {3, 0}};
    std::vector<double> real_vec;
    ngraph::he::complex_vec_to_real_vec(real_vec, complex_vec);

    EXPECT_EQ(real_vec.size(), 2 * complex_vec.size());
    EXPECT_TRUE(ngraph::test::he::all_close(real_vec,
                                            std::vector<double>{-2, -1, 3, 0}));
  }
}

TEST(he_util, real_vec_to_complex_vec) {
  {
    std::vector<std::complex<double>> complex_vec;
    std::vector<double> real_vec{1, 2, 3, 4};
    ngraph::he::real_vec_to_complex_vec(complex_vec, real_vec);

    EXPECT_EQ(complex_vec.size(), 2);
    EXPECT_TRUE(ngraph::test::he::all_close(
        complex_vec, std::vector<std::complex<double>>{{1, 2}, {3, 4}}));
  }

  {
    std::vector<std::complex<double>> complex_vec;
    std::vector<double> real_vec{-2, -1, 3};
    ngraph::he::real_vec_to_complex_vec(complex_vec, real_vec);

    EXPECT_EQ(complex_vec.size(), 2);
    EXPECT_TRUE(ngraph::test::he::all_close(
        complex_vec, std::vector<std::complex<double>>{{-2, -1}, {3, 0}}));
  }
}

TEST(he_util, flag_to_bool) {
  EXPECT_TRUE(ngraph::he::flag_to_bool(nullptr, true));
  EXPECT_FALSE(ngraph::he::flag_to_bool(nullptr, false));

  for (const auto& on_str : std::vector<std::string>{
           "1", "on", "ON", "y", "Y", "yes", "YES", "true", "TRUE"}) {
    EXPECT_TRUE(ngraph::he::flag_to_bool(on_str.c_str(), false));
  }

  for (const auto& off_str : std::vector<std::string>{
           "0", "off", "OFF", "n", "N", "no", "NO", "false", "FALSE"}) {
    EXPECT_FALSE(ngraph::he::flag_to_bool(off_str.c_str(), true));
  }

  EXPECT_ANY_THROW(ngraph::he::flag_to_bool("DUMMY_VAL"));
}

TEST(he_util, type_to_double) {
  auto test_type_to_double = [](auto x) {
    EXPECT_DOUBLE_EQ(
        static_cast<double>(x),
        ngraph::he::type_to_double(static_cast<double*>(static_cast<void*>(&x)),
                                   ngraph::element::from<decltype(x)>()));
  };

  test_type_to_double(double{10.7});
  test_type_to_double(float{10.3});
  test_type_to_double(int32_t{10});
  test_type_to_double(int64_t{10});

  // Unsupported type
  EXPECT_ANY_THROW(ngraph::he::type_to_double(nullptr, ngraph::element::i8));
}

TEST(he_util, param_originates_from_name) {
  ngraph::op::Parameter param{ngraph::element::f32, ngraph::Shape{}};

  EXPECT_TRUE(ngraph::he::param_originates_from_name(param, param.get_name()));
  EXPECT_FALSE(ngraph::he::param_originates_from_name(param, "wrong_name"));

  param.add_provenance_tag("tag");
  EXPECT_TRUE(ngraph::he::param_originates_from_name(param, "tag"));

  param.remove_provenance_tag("tag");
  EXPECT_FALSE(ngraph::he::param_originates_from_name(param, "tag"));
}

TEST(he_util, type_to_pb_type) {
  for (const auto& type : std::vector<ngraph::element::Type>{
           ngraph::element::Type_t::undefined, ngraph::element::Type_t::dynamic,
           ngraph::element::Type_t::boolean, ngraph::element::Type_t::bf16,
           ngraph::element::Type_t::f16, ngraph::element::Type_t::f32,
           ngraph::element::Type_t::f64, ngraph::element::Type_t::i8,
           ngraph::element::Type_t::i16, ngraph::element::Type_t::i32,
           ngraph::element::Type_t::i64, ngraph::element::Type_t::u8,
           ngraph::element::Type_t::u16, ngraph::element::Type_t::u32,
           ngraph::element::Type_t::u64}) {
    EXPECT_EQ(ngraph::he::pb_type_to_type(ngraph::he::type_to_pb_type(type)),
              type);
  }
}