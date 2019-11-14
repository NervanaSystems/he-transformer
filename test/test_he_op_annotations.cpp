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

#include "gtest/gtest.h"
#include "he_op_annotations.hpp"
#include "ngraph/op/parameter.hpp"

namespace ngraph::he {

TEST(he_op_annotations, set_get) {
  auto ann = HEOpAnnotations(false, false, false);
  EXPECT_FALSE(ann.from_client());
  EXPECT_FALSE(ann.encrypted());
  EXPECT_FALSE(ann.packed());

  ann.set_from_client(true);
  EXPECT_TRUE(ann.from_client());

  ann.set_from_client(false);
  EXPECT_FALSE(ann.from_client());

  ann.set_encrypted(true);
  EXPECT_TRUE(ann.encrypted());

  ann.set_encrypted(false);
  EXPECT_FALSE(ann.encrypted());

  ann.set_packed(true);
  EXPECT_TRUE(ann.packed());

  ann.set_packed(false);
  EXPECT_FALSE(ann.packed());
}

TEST(he_op_annotations, initialize) {
  HEOpAnnotations annotation{false, true, false};
  HEOpAnnotations annotation2{annotation};

  EXPECT_EQ(annotation, annotation2);
}

TEST(he_op_annotations, defaults) {
  auto param = std::make_shared<op::Parameter>(element::f32, Shape{});

  EXPECT_FALSE(HEOpAnnotations::from_client(*param));
  EXPECT_FALSE(HEOpAnnotations::plaintext_packed(*param));
}

}  // namespace ngraph::he
