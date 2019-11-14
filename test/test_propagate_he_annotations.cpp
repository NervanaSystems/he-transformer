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
#include "he_op_annotations.hpp"
#include "he_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/type/element_type.hpp"
#include "pass/propagate_he_annotations.hpp"
#include "test_util.hpp"
#include "util/test_tools.hpp"

namespace ngraph::he {

TEST(propagate_he_annotations, cipher) {
  Shape shape{2, 2};

  auto a = std::make_shared<op::Parameter>(element::f32, shape);
  auto b = std::make_shared<op::Parameter>(element::f32, shape);
  auto t = std::make_shared<op::Add>(a, b);
  auto f = std::make_shared<Function>(t, ParameterVector{a, b});

  bool arg1_enc = false;
  bool arg2_enc = true;
  bool packed = false;
  bool from_client = false;

  a->set_op_annotations(
      test::annotation_from_flags(from_client, arg1_enc, packed));
  b->set_op_annotations(
      test::annotation_from_flags(from_client, arg2_enc, packed));

  pass::PropagateHEAnnotations().run_on_function(f);

  auto a_annotation =
      std::dynamic_pointer_cast<HEOpAnnotations>(a->get_op_annotations());
  auto b_annotation =
      std::dynamic_pointer_cast<HEOpAnnotations>(b->get_op_annotations());
  auto t_annotation =
      std::dynamic_pointer_cast<HEOpAnnotations>(t->get_op_annotations());

  EXPECT_TRUE(a_annotation != nullptr);
  EXPECT_TRUE(b_annotation != nullptr);
  EXPECT_TRUE(t_annotation != nullptr);

  EXPECT_FALSE(a_annotation->encrypted());
  EXPECT_TRUE(b_annotation->encrypted());
  EXPECT_TRUE(t_annotation->encrypted());
}

TEST(propagate_he_annotations, pack) {
  Shape shape{2, 2};

  auto a = std::make_shared<op::Parameter>(element::f32, shape);
  auto b = std::make_shared<op::Parameter>(element::f32, shape);
  auto t = std::make_shared<op::Add>(a, b);
  auto f = std::make_shared<Function>(t, ParameterVector{a, b});

  bool arg1_packed = false;
  bool arg2_packed = true;
  bool encrypted = false;
  bool from_client = false;

  a->set_op_annotations(
      test::annotation_from_flags(from_client, encrypted, arg1_packed));
  b->set_op_annotations(
      test::annotation_from_flags(from_client, encrypted, arg2_packed));

  pass::PropagateHEAnnotations().run_on_function(f);

  auto a_annotation =
      std::dynamic_pointer_cast<HEOpAnnotations>(a->get_op_annotations());
  auto b_annotation =
      std::dynamic_pointer_cast<HEOpAnnotations>(b->get_op_annotations());
  auto t_annotation =
      std::dynamic_pointer_cast<HEOpAnnotations>(t->get_op_annotations());

  EXPECT_TRUE(a_annotation != nullptr);
  EXPECT_TRUE(b_annotation != nullptr);
  EXPECT_TRUE(t_annotation != nullptr);

  EXPECT_FALSE(a_annotation->packed());
  EXPECT_TRUE(b_annotation->packed());
  EXPECT_TRUE(t_annotation->packed());
}

}  // namespace ngraph::he
