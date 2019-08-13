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

using namespace std;
using namespace ngraph;

TEST(tcp_message, save_cipher) {
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
  CKKSEncoder ckks_encoder(context);

  vector<double> in_vals{0.0, 1.1, 2.2, 3.3};

  Plaintext plain;
  double scale = pow(2.0, 40);
  ckks_encoder.encode(in_vals, scale, plain);

  Ciphertext encrypted;
  encryptor.encrypt(plain, encrypted);

  size_t n = 100;
  std::vector<seal::Ciphertext> seal_ciphers(n);
  for (size_t i = 0; i < n; ++i) {
    seal_ciphers[i] = encrypted;
  }

  ngraph::he::TCPMessage message(ngraph::he::MessageType::execute,
                                 seal_ciphers);

  EXPECT_EQ(message.count(), n);

  for (size_t i = 0; i < n; ++i) {
    seal::Ciphertext cipher;
    ngraph::he::HEPlaintext plain;
    message.load_cipher(cipher, i, context);
    ngraph::he::decrypt(plain, cipher, false, decryptor, ckks_encoder);

    auto out_vals = plain.values();
    out_vals = std::vector<double>{out_vals.begin(),
                                   out_vals.begin() + in_vals.size()};

    EXPECT_TRUE(test::all_close(out_vals, in_vals, 1e-3, 1e-3));
  }
}

TEST(seal_util, save) {
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

  void* buffer = ngraph::ngraph_malloc(ngraph::he::ciphertext_size(cipher));
  ngraph::he::save(cipher, buffer);

  Ciphertext cipher_load;

  auto print_cipher = [](seal::Ciphertext& cipher) {
    NGRAPH_INFO << "ntt_form " << cipher.is_ntt_form();
    NGRAPH_INFO << "scale " << cipher.scale();
  };

  NGRAPH_INFO << "Encrypted";
  print_cipher(cipher);

  ngraph::he::load(cipher_load, context, buffer);

  NGRAPH_INFO << "Loaded";
  print_cipher(cipher_load);

  EXPECT_EQ(cipher_load.parms_id(), cipher.parms_id());
  EXPECT_EQ(cipher_load.is_ntt_form(), cipher.is_ntt_form());
  EXPECT_EQ(cipher_load.size(), cipher.size());
  EXPECT_EQ(cipher_load.poly_modulus_degree(), cipher.poly_modulus_degree());
  EXPECT_EQ(cipher_load.coeff_mod_count(), cipher.coeff_mod_count());
  EXPECT_EQ(cipher_load.scale(), cipher.scale());
  EXPECT_EQ(cipher_load.uint64_count(), cipher.uint64_count());
  EXPECT_EQ(cipher_load.is_transparent(), cipher.is_transparent());

  ngraph::ngraph_free(buffer);
}
