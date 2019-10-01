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

#include <google/protobuf/util/message_differencer.h>
#include <chrono>
#include <memory>

#include "gtest/gtest.h"
#include "protos/message.pb.h"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_cipher_tensor.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "tcp/tcp_message.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::he;

TEST(protobuf, trivial) { EXPECT_EQ(1, 1); }

TEST(protobuf, serialize_cipher) {
  he_proto::TCPMessage message;

  he_proto::Function f;
  f.set_function("123");
  *message.mutable_function() = f;

  std::stringstream s;
  message.SerializeToOstream(&s);

  he_proto::TCPMessage deserialize;
  deserialize.ParseFromIstream(&s);

  EXPECT_TRUE(
      google::protobuf::util::MessageDifferencer::Equals(deserialize, message));
}

TEST(tcp_message, create) {
  he_proto::TCPMessage proto_msg;
  he_proto::Function f;
  f.set_function("123");
  *proto_msg.mutable_function() = f;
  std::stringstream s;
  proto_msg.SerializeToOstream(&s);
  ngraph::he::TCPMessage tcp_message(std::move(proto_msg));
  EXPECT_EQ(1, 1);
}

TEST(tcp_message, encode_decode) {
  using data_buffer = std::vector<char>;
  data_buffer buffer;
  buffer.resize(20);

  size_t encode_size = 10;
  ngraph::he::TCPMessage::encode_header(buffer, encode_size);
  size_t decoded_size = ngraph::he::TCPMessage::decode_header(buffer);
  EXPECT_EQ(decoded_size, encode_size);
}

TEST(tcp_message, pack_unpack) {
  using data_buffer = std::vector<char>;

  he_proto::TCPMessage proto_msg;
  he_proto::Function f;
  f.set_function("123");
  *proto_msg.mutable_function() = f;
  std::stringstream s;
  proto_msg.SerializeToOstream(&s);
  ngraph::he::TCPMessage message1(std::move(proto_msg));

  data_buffer buffer;
  message1.pack(buffer);

  ngraph::he::TCPMessage message2;
  message2.unpack(buffer);

  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      *message1.proto_message(), *message2.proto_message()));
}

TEST(seal_cipher_wrapper, load_save) {
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
  double scale = pow(2.0, 40);
  encoder.encode(input, scale, plain);
  seal::Ciphertext c;
  encryptor.encrypt(plain, c);
  std::stringstream ss_save;
  c.save(ss_save);

  he_proto::SealCiphertextWrapper proto_cipher;

  ngraph::he::SealCiphertextWrapper cipher;
  cipher.ciphertext() = c;
  cipher.complex_packing() = true;
  cipher.known_value() = false;

  typedef std::chrono::high_resolution_clock Clock;
  auto t1 = Clock::now();
  cipher.save(proto_cipher);
  auto t2 = Clock::now();
  NGRAPH_INFO
      << "Save time "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << "us";

  std::shared_ptr<ngraph::he::SealCiphertextWrapper> cipher_load;
  auto t3 = Clock::now();
  ngraph::he::SealCiphertextWrapper::load(cipher_load, proto_cipher, context);
  auto t4 = Clock::now();
  NGRAPH_INFO
      << "Load time "
      << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count()
      << "us";

  std::stringstream ss_load;
  cipher_load->ciphertext().save(ss_load);

  EXPECT_EQ(cipher.complex_packing(), cipher_load->complex_packing());
  EXPECT_EQ(cipher.known_value(), cipher_load->known_value());
  EXPECT_EQ(ss_save.str(), ss_load.str());
}

TEST(seal_cipher_wrapper, load_save_known_value) {
  using namespace seal;
  EncryptionParameters parms(scheme_type::CKKS);
  size_t poly_modulus_degree = 8192;
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(
      CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
  auto context = SEALContext::Create(parms);

  he_proto::SealCiphertextWrapper proto_cipher;

  ngraph::he::SealCiphertextWrapper cipher;
  cipher.ciphertext() = seal::Ciphertext();
  cipher.complex_packing() = false;
  cipher.known_value() = true;
  cipher.value() = 1.23;

  typedef std::chrono::high_resolution_clock Clock;
  auto t1 = Clock::now();
  cipher.save(proto_cipher);
  auto t2 = Clock::now();
  NGRAPH_INFO
      << "Save time "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << "us";

  std::shared_ptr<ngraph::he::SealCiphertextWrapper> cipher_load;
  auto t3 = Clock::now();
  ngraph::he::SealCiphertextWrapper::load(cipher_load, proto_cipher, context);
  auto t4 = Clock::now();
  NGRAPH_INFO
      << "Load time "
      << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count()
      << "us";

  EXPECT_EQ(cipher.complex_packing(), cipher_load->complex_packing());
  EXPECT_EQ(cipher.known_value(), cipher_load->known_value());
  EXPECT_EQ(cipher.value(), cipher_load->value());
}

TEST(seal_cipher_tensor, load_save) {
  auto backend = runtime::Backend::create("HE_SEAL");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  auto parms =
      ngraph::he::HESealEncryptionParameters::default_real_packing_parms();
  he_backend->update_encryption_parameters(parms);

  Shape shape{2};
  auto a = he_backend->create_cipher_tensor(element::f32, shape);
  copy_data(a, vector<float>{5, 6});

  auto he_tensor = dynamic_pointer_cast<ngraph::he::HESealCipherTensor>(a);
  EXPECT_TRUE(he_tensor != nullptr);

  std::vector<he_proto::SealCipherTensor> protos;

  he_tensor->save_to_proto(protos);

  EXPECT_EQ(protos.size(), 1);
  EXPECT_EQ(protos[0].name() == he_tensor->get_name());
}