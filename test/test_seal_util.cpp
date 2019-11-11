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

#include <sstream>
#include <unordered_set>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "seal/seal.h"
#include "seal/seal_util.hpp"
#include "test_util.hpp"
#include "util/test_tools.hpp"

TEST(seal_util, seal_security_level) {
  EXPECT_EQ(ngraph::he::seal_security_level(0), seal::sec_level_type::none);
  EXPECT_EQ(ngraph::he::seal_security_level(128), seal::sec_level_type::tc128);
  EXPECT_EQ(ngraph::he::seal_security_level(192), seal::sec_level_type::tc192);
  EXPECT_EQ(ngraph::he::seal_security_level(256), seal::sec_level_type::tc256);

  EXPECT_ANY_THROW({ ngraph::he::seal_security_level(42); });
}

TEST(seal_util, save) {
  seal::EncryptionParameters parms(seal::scheme_type::CKKS);
  size_t poly_modulus_degree = 8192;
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(
      seal::CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));

  auto context = seal::SEALContext::Create(parms);

  seal::KeyGenerator keygen(context);
  auto public_key = keygen.public_key();
  auto secret_key = keygen.secret_key();
  auto relin_keys = keygen.relin_keys();

  seal::Encryptor encryptor(context, public_key);
  seal::Evaluator evaluator(context);
  seal::Decryptor decryptor(context, secret_key);
  seal::CKKSEncoder encoder(context);

  std::vector<double> input{0.0, 1.1, 2.2, 3.3};

  seal::Plaintext plain;
  double scale = pow(2.0, 60);
  encoder.encode(input, scale, plain);

  seal::Ciphertext cipher;
  encryptor.encrypt(plain, cipher);
  seal::Ciphertext cipher_load;

  auto* buffer = reinterpret_cast<std::byte*>(
      ngraph::ngraph_malloc(ngraph::he::ciphertext_size(cipher)));

  auto t1 = std::chrono::high_resolution_clock::now();
  auto save_size = ngraph::he::save(cipher, buffer);
  auto t2 = std::chrono::high_resolution_clock::now();
  ngraph::he::load(cipher_load, context, buffer, save_size);
  auto t3 = std::chrono::high_resolution_clock::now();

  NGRAPH_INFO
      << "save time "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << "us";
  NGRAPH_INFO
      << "load time "
      << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count()
      << "us";

  EXPECT_EQ(cipher_load.parms_id(), cipher.parms_id());
  EXPECT_EQ(cipher_load.is_ntt_form(), cipher.is_ntt_form());
  EXPECT_EQ(cipher_load.size(), cipher.size());
  EXPECT_EQ(cipher_load.poly_modulus_degree(), cipher.poly_modulus_degree());
  EXPECT_EQ(cipher_load.coeff_mod_count(), cipher.coeff_mod_count());
  EXPECT_EQ(cipher_load.scale(), cipher.scale());
  EXPECT_EQ(cipher_load.is_transparent(), cipher.is_transparent());

  for (size_t i = 0; i < cipher.int_array().size(); ++i) {
    EXPECT_EQ(cipher_load[i], cipher[i]);
  }
  ngraph::ngraph_free(buffer);
}
