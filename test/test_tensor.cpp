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

#include "gtest/gtest.h"
#include "he_plain_tensor.hpp"
#include "he_tensor.hpp"
#include "ngraph/ngraph.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_cipher_tensor.hpp"
#include "seal/he_seal_executable.hpp"
#include "test_util.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::he;

TEST(he_tensor, pack) {
  Shape shape{2, 2};

  HEPlainTensor plain(element::f32, shape, false);

  std::vector<HEPlaintext> elements;
  for (size_t i = 0; i < shape_size(shape); ++i) {
    elements.push_back(HEPlaintext(i));
  }
  plain.set_elements(elements);
  plain.pack();

  EXPECT_TRUE(plain.is_packed());
  EXPECT_EQ(plain.get_batch_size(), 2);
  EXPECT_EQ(plain.num_plaintexts(), 2);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(plain.get_element(i).num_values(), 2);
  }
  EXPECT_EQ(plain.get_element(0).values()[0], 0);
  EXPECT_EQ(plain.get_element(0).values()[1], 2);
  EXPECT_EQ(plain.get_element(1).values()[0], 1);
  EXPECT_EQ(plain.get_element(1).values()[1], 3);
}

TEST(he_tensor, unpack) {
  Shape shape{2, 2};

  HEPlainTensor plain(element::f32, shape, true);

  std::vector<HEPlaintext> elements;
  elements.push_back(HEPlaintext(std::vector<double>{0, 1}));
  elements.push_back(HEPlaintext(std::vector<double>{2, 3}));
  plain.set_elements(elements);

  plain.unpack();

  EXPECT_FALSE(plain.is_packed());
  EXPECT_EQ(plain.num_plaintexts(), 4);
  EXPECT_EQ(plain.get_batch_size(), 1);

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(plain.get_element(i).num_values(), 1);
  }
  EXPECT_EQ(plain.get_element(0).values()[0], 0);
  EXPECT_EQ(plain.get_element(1).values()[0], 2);
  EXPECT_EQ(plain.get_element(2).values()[0], 1);
  EXPECT_EQ(plain.get_element(3).values()[0], 3);
}
