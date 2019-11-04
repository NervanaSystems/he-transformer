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
#include "seal/seal.h"
#include "seal/seal_util.hpp"
#include "tcp/tcp_message.hpp"
#include "util/all_close.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

TEST(seal_example, trivial) {
  int a = 1;
  int b = 2;
  EXPECT_EQ(3, a + b);
}

TEST(seal_example, seal_ckks_basics) {

  EncryptionParameters parms(scheme_type::CKKS);
  size_t poly_modulus_degree = 8192;
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(
      CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));

  auto context = SEALContext::Create(parms);
  // print_parameters(context);

  KeyGenerator keygen(context);
  auto public_key = keygen.public_key();
  auto secret_key = keygen.secret_key();
  auto relin_keys = keygen.relin_keys();

  Encryptor encryptor(context, public_key);
  Evaluator evaluator(context);
  Decryptor decryptor(context, secret_key);
  CKKSEncoder encoder(context);

  vector<double> input{0.0, 1.1, 2.2, 3.3};

  Plaintext plain;
  double scale = pow(2.0, 40);
  encoder.encode(input, scale, plain);

  Ciphertext encrypted;
  encryptor.encrypt(plain, encrypted);

  evaluator.square_inplace(encrypted);
  evaluator.relinearize_inplace(encrypted, relin_keys);
  decryptor.decrypt(encrypted, plain);
  encoder.decode(plain, input);

  evaluator.mod_switch_to_next_inplace(encrypted);

  decryptor.decrypt(encrypted, plain);

  encoder.decode(plain, input);

  encrypted.scale() *= 3;
  decryptor.decrypt(encrypted, plain);
  encoder.decode(plain, input);
}

TEST(seal_example, seal_ckks_complex_conjugate) {
  using namespace seal;

  EncryptionParameters parms(scheme_type::CKKS);
  size_t poly_modulus_degree = 8192;
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(
      CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));

  auto context = SEALContext::Create(parms);
  // print_parameters(context);

  KeyGenerator keygen(context);
  auto public_key = keygen.public_key();
  auto secret_key = keygen.secret_key();
  auto relin_keys = keygen.relin_keys();
  auto galois_keys = keygen.galois_keys();

  Encryptor encryptor(context, public_key);
  Evaluator evaluator(context);
  Decryptor decryptor(context, secret_key);
  CKKSEncoder encoder(context);

  vector<complex<double>> input{{0.0, 1.1}, {2.2, 3.3}};
  vector<complex<double>> exp_output{{0.0, -1.1}, {2.2, -3.3}};
  vector<complex<double>> output;

  Plaintext plain;
  double scale = pow(2.0, 40);
  encoder.encode(input, scale, plain);

  Ciphertext encrypted;
  encryptor.encrypt(plain, encrypted);
  evaluator.complex_conjugate_inplace(encrypted, galois_keys);

  decryptor.decrypt(encrypted, plain);
  encoder.decode(plain, output);

  EXPECT_TRUE(abs(exp_output[0] - output[0]) < 0.1);
  EXPECT_TRUE(abs(exp_output[1] - output[1]) < 0.1);
}

TEST(seal_util, save) {
  using namespace seal;

  EncryptionParameters parms(scheme_type::CKKS);
  size_t poly_modulus_degree = 8192;
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(
      CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));

  auto context = SEALContext::Create(parms);

  KeyGenerator keygen(context);
  auto public_key = keygen.public_key();
  auto secret_key = keygen.secret_key();
  auto relin_keys = keygen.relin_keys();

  Encryptor encryptor(context, public_key);
  Evaluator evaluator(context);
  Decryptor decryptor(context, secret_key);
  CKKSEncoder encoder(context);

  vector<double> input{0.0, 1.1, 2.2, 3.3};

  Plaintext plain;
  double scale = pow(2.0, 60);
  encoder.encode(input, scale, plain);

  Ciphertext cipher;
  encryptor.encrypt(plain, cipher);
  Ciphertext cipher_load;

  auto* buffer = reinterpret_cast<std::byte*>(
      ngraph::ngraph_malloc(ngraph::he::ciphertext_size(cipher)));

  auto t1 = chrono::high_resolution_clock::now();
  auto save_size = ngraph::he::save(cipher, buffer);
  auto t2 = chrono::high_resolution_clock::now();
  ngraph::he::load(cipher_load, context, buffer, save_size);
  auto t3 = chrono::high_resolution_clock::now();

  NGRAPH_INFO << "save time "
              << chrono::duration_cast<chrono::microseconds>(t2 - t1).count()
              << "us";
  NGRAPH_INFO << "load time "
              << chrono::duration_cast<chrono::microseconds>(t3 - t2).count()
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
