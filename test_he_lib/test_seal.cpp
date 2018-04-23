/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include<memory>

#include "gtest/gtest.h"
#include "seal/seal.h"

using namespace std;

TEST(test_he, trivial)
{
    int a = 1;
    int b = 2;
    EXPECT_EQ(3, a + b);
}

TEST(seal_example, basics_i)
{
    using namespace seal;

    // Parameter
    EncryptionParameters parms;
    parms.set_poly_modulus("1x^2048 + 1");
    parms.set_coeff_modulus(coeff_modulus_128(2048));
    parms.set_plain_modulus(1 << 8);

    // Context: print with print_parameters(context);
    SEALContext context(parms);

    // Objects from context
    IntegerEncoder encoder(context.plain_modulus());
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

    // Compute
    evaluator.negate(encrypted1);
    evaluator.add(encrypted1, encrypted2);
    evaluator.multiply(encrypted1, encrypted2);

    // Decrypt
    Plaintext plain_result;
    decryptor.decrypt(encrypted1, plain_result);

    // Decode
    int result = encoder.decode_int32(plain_result);
    EXPECT_EQ(84, result);
}

TEST(seal_example, shared_ptr_encrypt)
{
    using namespace seal;

    // Parameter
    EncryptionParameters parms;
    parms.set_poly_modulus("1x^2048 + 1");
    parms.set_coeff_modulus(coeff_modulus_128(2048));
    parms.set_plain_modulus(1 << 8);

    // Context: print with print_parameters(context);
    SEALContext context(parms);

    // Objects from context
    IntegerEncoder encoder(context.plain_modulus());
    KeyGenerator keygen(context);
    PublicKey public_key = keygen.public_key();
    SecretKey secret_key = keygen.secret_key();
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    // Encode
    int value1 = 5;
    Plaintext plain = encoder.encode(value1);

    // Encrypt
    auto encrypted_ptr = make_shared<Ciphertext>();
    encryptor.encrypt(plain, *encrypted_ptr);

    // Decrypt
    Plaintext plain_result;
    decryptor.decrypt(*encrypted_ptr, plain_result);

    // Decode
    int result = encoder.decode_int32(plain_result);
    EXPECT_EQ(5, result);
}
