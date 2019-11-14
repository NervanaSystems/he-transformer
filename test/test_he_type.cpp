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
#include "he_plaintext.hpp"
#include "he_type.hpp"
#include "ngraph/ngraph.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "test_util.hpp"
#include "util/test_tools.hpp"

namespace ngraph::runtime::he {

TEST(he_type, plain) {
  {
    HEPlaintext plain{1, 2, 3};
    bool complex_packing = false;
    auto he_type = HEType(plain, complex_packing);
    EXPECT_TRUE(he_type.is_plaintext());
    EXPECT_FALSE(he_type.is_ciphertext());
    EXPECT_FALSE(he_type.complex_packing());
    EXPECT_TRUE(he_type.plaintext_packing());
    EXPECT_EQ(he_type.batch_size(), 3);
    EXPECT_TRUE(test::all_close(plain, he_type.get_plaintext()));

    he_type.set_ciphertext(nullptr);
    EXPECT_FALSE(he_type.is_plaintext());
    EXPECT_TRUE(he_type.is_ciphertext());
  }
  {
    HEPlaintext plain{1};
    bool complex_packing = true;
    auto he_type = HEType(plain, complex_packing);
    EXPECT_TRUE(he_type.is_plaintext());
    EXPECT_FALSE(he_type.is_ciphertext());
    EXPECT_TRUE(he_type.complex_packing());
    EXPECT_FALSE(he_type.plaintext_packing());
    EXPECT_EQ(he_type.batch_size(), 1);
    EXPECT_TRUE(test::all_close(plain, he_type.get_plaintext()));

    he_type.set_ciphertext(nullptr);
    EXPECT_FALSE(he_type.is_plaintext());
    EXPECT_TRUE(he_type.is_ciphertext());
  }
}

TEST(he_type, cipher) {
  {
    size_t batch_size = 10;
    bool complex_packing = false;
    auto cipher = HESealBackend::create_empty_ciphertext();
    auto he_type = HEType(cipher, complex_packing, batch_size);

    EXPECT_EQ(he_type.batch_size(), batch_size);
    EXPECT_EQ(he_type.complex_packing(), complex_packing);
    EXPECT_TRUE(he_type.is_ciphertext());
    EXPECT_TRUE(he_type.plaintext_packing());
    EXPECT_FALSE(he_type.is_plaintext());

    he_type.set_plaintext(HEPlaintext());

    EXPECT_FALSE(he_type.is_ciphertext());
    EXPECT_TRUE(he_type.is_plaintext());
  }
  {
    size_t batch_size = 27;
    bool complex_packing = true;
    auto cipher = HESealBackend::create_empty_ciphertext();
    auto he_type = HEType(cipher, complex_packing, batch_size);

    EXPECT_EQ(he_type.batch_size(), batch_size);
    EXPECT_EQ(he_type.complex_packing(), complex_packing);
    EXPECT_TRUE(he_type.is_ciphertext());
    EXPECT_TRUE(he_type.plaintext_packing());
    EXPECT_FALSE(he_type.is_plaintext());

    he_type.set_plaintext(HEPlaintext());

    EXPECT_FALSE(he_type.is_ciphertext());
    EXPECT_TRUE(he_type.is_plaintext());
  }
}

TEST(he_type, save_load) {
  {
    HEPlaintext plain{1, 2, 3};
    bool complex_packing = false;
    auto he_type = HEType(plain, complex_packing);

    pb::HEType proto_type;

    he_type.save(proto_type);

    EXPECT_EQ(proto_type.is_plaintext(), he_type.is_plaintext());
    EXPECT_EQ(proto_type.plaintext_packing(), he_type.plaintext_packing());
    EXPECT_EQ(proto_type.complex_packing(), he_type.complex_packing());
    EXPECT_EQ(proto_type.batch_size(), he_type.batch_size());

    auto loaded_he_type = HEType::load(proto_type, nullptr);
    EXPECT_EQ(loaded_he_type.is_plaintext(), he_type.is_plaintext());
    EXPECT_EQ(loaded_he_type.plaintext_packing(), he_type.plaintext_packing());
    EXPECT_EQ(loaded_he_type.complex_packing(), he_type.complex_packing());
    EXPECT_EQ(loaded_he_type.batch_size(), he_type.batch_size());

    EXPECT_TRUE(test::all_close(loaded_he_type.get_plaintext(),
                                            he_type.get_plaintext()));
  }
  {
    HEPlaintext plain{7};
    bool complex_packing = true;
    auto he_type = HEType(plain, complex_packing);

    pb::HEType proto_type;

    he_type.save(proto_type);

    EXPECT_EQ(proto_type.is_plaintext(), he_type.is_plaintext());
    EXPECT_EQ(proto_type.plaintext_packing(), he_type.plaintext_packing());
    EXPECT_EQ(proto_type.complex_packing(), he_type.complex_packing());
    EXPECT_EQ(proto_type.batch_size(), he_type.batch_size());

    auto loaded_he_type = HEType::load(proto_type, nullptr);
    EXPECT_EQ(loaded_he_type.is_plaintext(), he_type.is_plaintext());
    EXPECT_EQ(loaded_he_type.plaintext_packing(), he_type.plaintext_packing());
    EXPECT_EQ(loaded_he_type.complex_packing(), he_type.complex_packing());
    EXPECT_EQ(loaded_he_type.batch_size(), he_type.batch_size());

    EXPECT_TRUE(test::all_close(loaded_he_type.get_plaintext(),
                                            he_type.get_plaintext()));
  }
}

}  // namespace ngraph::runtime::he
