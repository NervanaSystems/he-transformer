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
#include "seal/seal.h"

using namespace std;

TEST(seal_example, trivial) {
  int a = 1;
  int b = 2;
  EXPECT_EQ(3, a + b);
}

TEST(seal_example, seal_ckks_basics_i) {
  using namespace seal;

  EncryptionParameters parms(scheme_type::CKKS);
  parms.set_poly_modulus_degree(8192);
  parms.set_coeff_modulus(DefaultParams::coeff_modulus_128(8192));

  auto context = SEALContext::Create(parms);
  // print_parameters(context);

  KeyGenerator keygen(context);
  auto public_key = keygen.public_key();
  auto secret_key = keygen.secret_key();
  auto relin_keys = keygen.relin_keys(60);

  Encryptor encryptor(context, public_key);
  Evaluator evaluator(context);
  Decryptor decryptor(context, secret_key);
  CKKSEncoder encoder(context);

  vector<double> input{0.0, 1.1, 2.2, 3.3};

  Plaintext plain;
  double scale = pow(2.0, 60);
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

TEST(seal_example, seal_bfv_basics_i) {
  using namespace seal;

  // Parameter
  EncryptionParameters parms(seal::scheme_type::BFV);
  parms.set_poly_modulus_degree(2048);
  parms.set_coeff_modulus(DefaultParams::coeff_modulus_128(2048));
  parms.set_plain_modulus(1 << 8);

  // Context: print with print_parameters(context);
  auto context = SEALContext::Create(parms);

  // Objects from context
  IntegerEncoder encoder(context);
  KeyGenerator keygen(context);
  PublicKey public_key = keygen.public_key();
  SecretKey secret_key = keygen.secret_key();
  Encryptor encryptor(context, public_key);
  Evaluator evaluator(context);
  Decryptor decryptor(context, secret_key);

  // Encode
  int value1 = 5;
  Plaintext plain1 = encoder.encode(value1);
  int value2 = -7;
  Plaintext plain2 = encoder.encode(value2);

  // Encrypt
  Ciphertext encrypted1, encrypted2;

  encryptor.encrypt(plain1, encrypted1);
  encryptor.encrypt(plain2, encrypted2);

  // Compute. In-place add and multiply will over-write encrypted1
  evaluator.negate_inplace(encrypted1);
  evaluator.add_inplace(encrypted1, encrypted2);
  evaluator.multiply_inplace(encrypted1, encrypted2);

  // Decrypt
  Plaintext plain_result;
  decryptor.decrypt(encrypted1, plain_result);

  // Decode
  int result = encoder.decode_int32(plain_result);
  EXPECT_EQ(84, result);
}

TEST(seal_example, seal_ckks_test) {
  using namespace seal;

  EncryptionParameters parms(scheme_type::CKKS);
  parms.set_poly_modulus_degree(8192);
  parms.set_coeff_modulus(DefaultParams::coeff_modulus_128(8192));

  auto context = SEALContext::Create(parms);
  // print_parameters(context);

  KeyGenerator keygen(context);
  auto public_key = keygen.public_key();
  auto secret_key = keygen.secret_key();
  auto relin_keys = keygen.relin_keys(60);

  Encryptor encryptor(context, public_key);
  Evaluator evaluator(context);
  Decryptor decryptor(context, secret_key);
  CKKSEncoder encoder(context);

  // Test complex add
  // vector<double> input{1, 2, 3, 4};
  vector<complex<double>> input{{1, 2}, {3, 4}};

  Plaintext plain;
  double scale = pow(2.0, 60);
  encoder.encode(input, scale, plain);
  input.clear();
  Ciphertext encrypted;
  encryptor.encrypt(plain, encrypted);

  Plaintext plain2;
  complex<double> factor_val{{5, 6}};
  vector<complex<double>> factor{factor_val, factor_val};

  encoder.encode(factor, scale, plain2);

  evaluator.add_plain_inplace(encrypted, plain2);
  Plaintext plain3;
  decryptor.decrypt(encrypted, plain3);
  encoder.decode(plain3, input);
  std::cout << input[0] << ", " << input[1] << std::endl;
}

TEST(seal_example, seal_ckks_test2) {
  using namespace seal;

  EncryptionParameters parms(scheme_type::CKKS);
  parms.set_poly_modulus_degree(8192);
  parms.set_coeff_modulus(DefaultParams::coeff_modulus_128(8192));

  auto context = SEALContext::Create(parms);
  // print_parameters(context);

  KeyGenerator keygen(context);
  auto public_key = keygen.public_key();
  auto secret_key = keygen.secret_key();
  auto relin_keys = keygen.relin_keys(60);

  Encryptor encryptor(context, public_key);
  Evaluator evaluator(context);
  Decryptor decryptor(context, secret_key);
  CKKSEncoder encoder(context);

  // Test complex add
  // vector<double> input{1, 2, 3, 4};
  vector<complex<double>> input{{1, 2}, {3, 4}};

  Plaintext plain;
  double scale = pow(2.0, 60);
  encoder.encode(input, scale, plain);
  input.clear();
  Ciphertext encrypted;
  encryptor.encrypt(plain, encrypted);

  Plaintext plain2;
  double factor_val{5};
  vector<double> factor{factor_val, factor_val};

  encoder.encode(factor_val, scale, plain2);

  evaluator.add_plain_inplace(encrypted, plain2);
  Plaintext plain3;
  decryptor.decrypt(encrypted, plain3);
  encoder.decode(plain3, input);
  std::cout << input[0] << ", " << input[1] << std::endl;
}
