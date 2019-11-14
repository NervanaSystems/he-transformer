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

namespace ngraph::runtime::he {

TEST(he_util, complex_vec_to_real_vec) {
  {
    std::vector<std::complex<double>> complex_vec{{1, 2}, {3, 4}};
    std::vector<double> real_vec;
    complex_vec_to_real_vec(real_vec, complex_vec);

    EXPECT_EQ(real_vec.size(), 2 * complex_vec.size());
    EXPECT_TRUE(test::all_close(real_vec, std::vector<double>{1, 2, 3, 4}));
  }

  {
    std::vector<std::complex<double>> complex_vec{{-2, -1}, {3, 0}};
    std::vector<double> real_vec;
    complex_vec_to_real_vec(real_vec, complex_vec);

    EXPECT_EQ(real_vec.size(), 2 * complex_vec.size());
    EXPECT_TRUE(test::all_close(real_vec, std::vector<double>{-2, -1, 3, 0}));
  }
}

TEST(he_util, real_vec_to_complex_vec) {
  {
    std::vector<std::complex<double>> complex_vec;
    std::vector<double> real_vec{1, 2, 3, 4};
    real_vec_to_complex_vec(complex_vec, real_vec);

    EXPECT_EQ(complex_vec.size(), 2);
    EXPECT_TRUE(test::all_close(
        complex_vec, std::vector<std::complex<double>>{{1, 2}, {3, 4}}));
  }

  {
    std::vector<std::complex<double>> complex_vec;
    std::vector<double> real_vec{-2, -1, 3};
    real_vec_to_complex_vec(complex_vec, real_vec);

    EXPECT_EQ(complex_vec.size(), 2);
    EXPECT_TRUE(test::all_close(
        complex_vec, std::vector<std::complex<double>>{{-2, -1}, {3, 0}}));
  }
}

TEST(he_util, flag_to_bool) {
  EXPECT_TRUE(flag_to_bool(nullptr, true));
  EXPECT_FALSE(flag_to_bool(nullptr, false));

  for (const auto& on_str : std::vector<std::string>{
           "1", "on", "ON", "y", "Y", "yes", "YES", "true", "TRUE"}) {
    EXPECT_TRUE(flag_to_bool(on_str.c_str(), false));
  }

  for (const auto& off_str : std::vector<std::string>{
           "0", "off", "OFF", "n", "N", "no", "NO", "false", "FALSE"}) {
    EXPECT_FALSE(flag_to_bool(off_str.c_str(), true));
  }

  EXPECT_ANY_THROW(flag_to_bool("DUMMY_VAL"));
}

TEST(he_util, type_to_double) {
  auto test_type_to_double = [](auto x) {
    EXPECT_DOUBLE_EQ(
        static_cast<double>(x),
        type_to_double(static_cast<double*>(static_cast<void*>(&x)),
                       element::from<decltype(x)>()));
  };

  test_type_to_double(double{10.7});
  test_type_to_double(float{10.3});
  test_type_to_double(int32_t{10});
  test_type_to_double(int64_t{10});

  // Unsupported type
  EXPECT_ANY_THROW(type_to_double(nullptr, element::i8));
}

TEST(he_util, param_originates_from_name) {
  op::Parameter param{element::f32, Shape{}};

  EXPECT_TRUE(param_originates_from_name(param, param.get_name()));
  EXPECT_FALSE(param_originates_from_name(param, "wrong_name"));

  param.add_provenance_tag("tag");
  EXPECT_TRUE(param_originates_from_name(param, "tag"));

  param.remove_provenance_tag("tag");
  EXPECT_FALSE(param_originates_from_name(param, "tag"));
}

TEST(he_util, type_to_pb_type) {
  for (const auto& type : std::vector<element::Type>{
           element::Type_t::undefined, element::Type_t::dynamic,
           element::Type_t::boolean, element::Type_t::bf16,
           element::Type_t::f16, element::Type_t::f32, element::Type_t::f64,
           element::Type_t::i8, element::Type_t::i16, element::Type_t::i32,
           element::Type_t::i64, element::Type_t::u8, element::Type_t::u16,
           element::Type_t::u32, element::Type_t::u64}) {
    EXPECT_EQ(pb_type_to_type(type_to_pb_type(type)), type);
  }
}
}  // namespace ngraph::runtime::he
