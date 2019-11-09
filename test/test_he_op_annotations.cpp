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
#include "seal/he_seal_backend.hpp"

TEST(he_op_annotations, defaults) {
  {
    auto ann1 =
        ngraph::he::HEOpAnnotations::server_plaintext_unpacked_annotation();

    EXPECT_FALSE(ann1->from_client());
    EXPECT_FALSE(ann1->encrypted());
    EXPECT_FALSE(ann1->packed());
  }
  {
    auto ann2 =
        ngraph::he::HEOpAnnotations::server_plaintext_packed_annotation();

    EXPECT_FALSE(ann2->from_client());
    EXPECT_FALSE(ann2->encrypted());
    EXPECT_TRUE(ann2->packed());
  }
  {
    auto ann3 =
        ngraph::he::HEOpAnnotations::server_ciphertext_unpacked_annotation();

    EXPECT_FALSE(ann3->from_client());
    EXPECT_TRUE(ann3->encrypted());
    EXPECT_FALSE(ann3->packed());
  }
  {
    auto ann4 =
        ngraph::he::HEOpAnnotations::server_ciphertext_packed_annotation();

    EXPECT_FALSE(ann4->from_client());
    EXPECT_TRUE(ann4->encrypted());
    EXPECT_TRUE(ann4->packed());
  }
  {
    auto ann5 =
        ngraph::he::HEOpAnnotations::client_plaintext_unpacked_annotation();

    EXPECT_TRUE(ann5->from_client());
    EXPECT_FALSE(ann5->encrypted());
    EXPECT_FALSE(ann5->packed());
  }
  {
    auto ann6 =
        ngraph::he::HEOpAnnotations::client_plaintext_packed_annotation();

    EXPECT_TRUE(ann6->from_client());
    EXPECT_FALSE(ann6->encrypted());
    EXPECT_TRUE(ann6->packed());
  }
  {
    auto ann7 =
        ngraph::he::HEOpAnnotations::client_ciphertext_unpacked_annotation();

    EXPECT_TRUE(ann7->from_client());
    EXPECT_TRUE(ann7->encrypted());
    EXPECT_FALSE(ann7->packed());
  }
  {
    auto ann8 =
        ngraph::he::HEOpAnnotations::client_ciphertext_packed_annotation();

    EXPECT_TRUE(ann8->from_client());
    EXPECT_TRUE(ann8->encrypted());
    EXPECT_TRUE(ann8->packed());
  }
}

TEST(he_op_annotations, set_get) {
  auto ann = ngraph::he::HEOpAnnotations(false, false, false);
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
