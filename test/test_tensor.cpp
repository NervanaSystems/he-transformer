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
#include "he_tensor.hpp"
#include "he_type.hpp"
#include "ngraph/ngraph.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_executable.hpp"
#include "test_util.hpp"
#include "util/test_tools.hpp"

TEST(he_tensor, pack) {
  auto backend = ngraph::runtime::Backend::create("HE_SEAL");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  ngraph::Shape shape{2, 2};
  ngraph::he::HETensor plain(ngraph::element::f32, shape, false, false, false,
                             *he_backend);

  std::vector<ngraph::he::HEType> elements;
  for (size_t i = 0; i < shape_size(shape); ++i) {
    elements.emplace_back(ngraph::he::HEPlaintext({static_cast<double>(i)}),
                          false);
  }
  plain.data() = elements;
  plain.pack();

  EXPECT_TRUE(plain.is_packed());
  EXPECT_EQ(plain.get_packed_shape(), (ngraph::Shape{1, 2}));
  EXPECT_EQ(plain.get_batch_size(), 2);
  EXPECT_EQ(plain.data().size(), 2);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_TRUE(plain.data(i).is_plaintext());
    EXPECT_EQ(plain.data(i).get_plaintext().size(), 2);
  }
  EXPECT_EQ(plain.data(0).get_plaintext()[0], 0);
  EXPECT_EQ(plain.data(0).get_plaintext()[1], 2);
  EXPECT_EQ(plain.data(1).get_plaintext()[0], 1);
  EXPECT_EQ(plain.data(1).get_plaintext()[1], 3);
}

TEST(he_tensor, unpack) {
  auto backend = ngraph::runtime::Backend::create("HE_SEAL");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  ngraph::Shape shape{2, 2};
  ngraph::he::HETensor plain(ngraph::element::f32, shape, true, false, false,
                             *he_backend);
  std::vector<ngraph::he::HEType> elements;

  elements.emplace_back(ngraph::he::HEPlaintext(std::vector<double>{0, 1}),
                        false);
  elements.emplace_back(ngraph::he::HEPlaintext(std::vector<double>{2, 3}),
                        false);
  plain.data() = elements;
  plain.unpack();

  EXPECT_FALSE(plain.is_packed());
  EXPECT_EQ(plain.get_packed_shape(), (ngraph::Shape{2, 2}));
  EXPECT_EQ(plain.data().size(), 4);
  EXPECT_EQ(plain.get_batch_size(), 1);

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_TRUE(plain.data(i).is_plaintext());
    EXPECT_EQ(plain.data(i).get_plaintext().size(), 1);
  }
  EXPECT_EQ(plain.data(0).get_plaintext()[0], 0);
  EXPECT_EQ(plain.data(1).get_plaintext()[0], 2);
  EXPECT_EQ(plain.data(2).get_plaintext()[0], 1);
  EXPECT_EQ(plain.data(3).get_plaintext()[0], 3);
}

TEST(he_tensor, save) {
  auto backend = ngraph::runtime::Backend::create("HE_SEAL");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  auto parms =
      ngraph::he::HESealEncryptionParameters::default_real_packing_parms();
  he_backend->update_encryption_parameters(parms);

  ngraph::Shape shape{2};

  auto tensor = he_backend->create_plain_tensor(ngraph::element::f32, shape);
  std::vector<float> tensor_data({5, 6});

  copy_data(tensor, tensor_data);
  auto he_tensor = std::static_pointer_cast<ngraph::he::HETensor>(tensor);

  std::vector<ngraph::he::proto::HETensor> protos;
  he_tensor->write_to_protos(protos);

  EXPECT_EQ(protos.size(), 1);
  const auto& proto = protos[0];
  EXPECT_EQ(proto.name(), he_tensor->get_name());

  std::vector<uint64_t> expected_shape{shape};
  for (size_t shape_idx = 0; shape_idx < expected_shape.size(); ++shape_idx) {
    EXPECT_EQ(proto.shape(shape_idx), expected_shape[shape_idx]);
  }

  EXPECT_EQ(proto.offset(), 0);
  EXPECT_EQ(proto.packed(), he_tensor->is_packed());
  EXPECT_EQ(proto.data_size(), he_tensor->data().size());
  for (size_t i = 0; i < he_tensor->data().size(); ++i) {
    EXPECT_TRUE(proto.data(i).is_plaintext());

    std::vector<float> plain = {proto.data(i).plain().begin(),
                                proto.data(i).plain().end()};
    EXPECT_EQ(plain.size(), 1);
    EXPECT_FLOAT_EQ(plain[0], tensor_data[i]);
  }
}
