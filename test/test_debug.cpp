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

#include "he_op_annotations.hpp"
#include "ngraph/ngraph.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/add_seal.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::he;

static string s_manifest = "${MANIFEST}";

TEST(add, mod_wrap) {
  auto backend = runtime::Backend::create("HE_SEAL");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  // Use
  // NGRAPH_HE_SEAL_CONFIG=$HE_TRANSFORMER/configs/he_seal_ckks_config_N11_L1_small.json

  // No modulus wrap
  for (size_t i = 1; i <= 1024; i += 1) {
    HEPlaintext plain(std::vector<double>(i, -24.5698));
    HEPlaintext mask(std::vector<double>(i, -31.4277));

    // No modulus wrap
    // HEPlaintext plain({-5.37476, -24.5698});
    // HEPlaintext mask({-31.4277, -31.4277});

    // No modulus wrap
    // HEPlaintext plain({-5.37476, -24.5698});
    // HEPlaintext mask({-31.4277, -31.4277});

    // No modulus wrapping
    // HEPlaintext plain({-5.37476, -24.5698});
    // HEPlaintext mask({5.37476, -31.4277});

    // Modulus wrapping as expected
    // HEPlaintext plain({-24.5698});
    // HEPlaintext mask({-31.4277});

    auto cipher = HESealBackend::create_empty_ciphertext();

    he_backend->encrypt(cipher, plain, element::f32);

    scalar_add_seal(*cipher, mask, cipher, false, *he_backend);

    he_backend->decrypt(plain, *cipher, false);

    plain = HEPlaintext(std::vector<double>{plain.begin(), plain.begin() + 2});

    NGRAPH_INFO << "i " << i << " plain " << plain[0];
  }
}

TEST(add, mod_wrap_seal) {
  using namespace seal;

  EncryptionParameters parms(scheme_type::CKKS);
  size_t poly_modulus_degree = 2048;
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {30}));

  auto context = SEALContext::Create(parms);
  KeyGenerator keygen(context);
  auto public_key = keygen.public_key();
  auto secret_key = keygen.secret_key();
  auto relin_keys = keygen.relin_keys();

  Encryptor encryptor(context, public_key);
  Evaluator evaluator(context);
  Decryptor decryptor(context, secret_key);
  CKKSEncoder encoder(context);

  Plaintext plain;
  Plaintext mask;
  Ciphertext cipher;

  double scale = 1 << 24;

  for (size_t i = 1; i <= 1024; i += 1) {
    std::vector<double> plain_val(i, -24.5698);
    std::vector<double> mask_val(i, -31.4277);

    // Bad case
    {
      for (size_t j = 1; j < i; ++j) {
        plain_val[j] = 24;
      }
      encoder.encode(plain_val, scale, plain);
      encoder.encode(mask_val, scale, mask);
      encryptor.encrypt(plain, cipher);
      evaluator.add_plain_inplace(cipher, mask);
      decryptor.decrypt(cipher, plain);
      encoder.decode(plain, plain_val);
      NGRAPH_INFO << "i " << i << " plain bad " << plain_val[0];
    }
    // Good case
    {
      plain_val = std::vector<double>(i, -24.5698);
      encoder.encode(plain_val, scale, plain);
      encoder.encode(mask_val, scale, mask);
      encryptor.encrypt(plain, cipher);
      evaluator.add_plain_inplace(cipher, mask);
      decryptor.decrypt(cipher, plain);
      encoder.decode(plain, plain_val);
      NGRAPH_INFO << "i " << i << " plain good " << plain_val[0];
    }
  }
}