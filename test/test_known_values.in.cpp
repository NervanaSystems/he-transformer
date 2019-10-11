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
#include <vector>

#include "ngraph/ngraph.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/add_seal.hpp"
#include "seal/kernel/bounded_relu_seal.hpp"
#include "seal/kernel/multiply_seal.hpp"
#include "seal/kernel/negate_seal.hpp"
#include "seal/kernel/relu_seal.hpp"
#include "seal/kernel/subtract_seal.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::he;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, known_cipher_known_cipher_add) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  size_t count = 10;

  vector<shared_ptr<SealCiphertextWrapper>> arg0;
  vector<shared_ptr<SealCiphertextWrapper>> arg1;
  vector<shared_ptr<SealCiphertextWrapper>> out;
  vector<float> exp_out(count);

  for (size_t i = 0; i < count; i++) {
    arg0.emplace_back(make_shared<SealCiphertextWrapper>());
    arg1.emplace_back(make_shared<SealCiphertextWrapper>());
    out.emplace_back(make_shared<SealCiphertextWrapper>());

    arg0[i]->known_value() = true;
    arg0[i]->value() = i;

    arg1[i]->known_value() = true;
    arg1[i]->value() = i * 1.23;

    exp_out[i] = arg0[i]->value() + arg1[i]->value();
  }
  add_seal(arg0, arg1, out, ngraph::element::f32, *he_backend, count);

  for (size_t i = 0; i < count; ++i) {
    auto c_out = out[i];
    EXPECT_TRUE(c_out->known_value());
    EXPECT_EQ(c_out->value(), exp_out[i]);
  }
}

NGRAPH_TEST(${BACKEND_NAME}, known_cipher_cipher_add) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  size_t count = 10;

  vector<shared_ptr<SealCiphertextWrapper>> arg0;
  vector<shared_ptr<SealCiphertextWrapper>> arg1;
  vector<shared_ptr<SealCiphertextWrapper>> out;
  vector<float> exp_out(count);

  for (size_t i = 0; i < count; i++) {
    arg0.emplace_back(make_shared<SealCiphertextWrapper>());
    arg1.emplace_back(make_shared<SealCiphertextWrapper>());
    out.emplace_back(make_shared<SealCiphertextWrapper>());

    he_backend->encrypt(arg0[i], HEPlaintext(i), ngraph::element::f32, false);
    arg0[i]->known_value() = false;

    arg1[i]->known_value() = true;
    arg1[i]->value() = i * 1.23;

    exp_out[i] = i + arg1[i]->value();
  }
  add_seal(arg0, arg1, out, ngraph::element::f32, *he_backend, count);

  for (size_t i = 0; i < count; ++i) {
    EXPECT_FALSE(out[i]->known_value());
    HEPlaintext p;
    he_backend->decrypt(p, *out[i]);
    EXPECT_NEAR(p.values()[0], exp_out[i], 1e-3f);
  }
}

NGRAPH_TEST(${BACKEND_NAME}, known_cipher_plain_add) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  size_t count = 10;

  vector<shared_ptr<SealCiphertextWrapper>> arg0;
  vector<HEPlaintext> arg1;
  vector<shared_ptr<SealCiphertextWrapper>> out;
  vector<float> exp_out(count);

  for (size_t i = 0; i < count; i++) {
    float arg1_val = i * 1.23f;
    arg0.emplace_back(make_shared<SealCiphertextWrapper>());
    arg1.emplace_back(HEPlaintext(arg1_val));
    out.emplace_back(make_shared<SealCiphertextWrapper>());

    arg0[i]->known_value() = true;
    arg0[i]->value() = i;

    exp_out[i] = arg0[i]->value() + arg1_val;
  }
  add_seal(arg0, arg1, out, ngraph::element::f32, *he_backend, count);

  for (size_t i = 0; i < count; ++i) {
    auto c_out = out[i];
    EXPECT_TRUE(c_out->known_value());
    EXPECT_EQ(c_out->value(), exp_out[i]);
  }
}

NGRAPH_TEST(${BACKEND_NAME}, known_cipher_known_cipher_mult) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  size_t count = 10;

  vector<shared_ptr<SealCiphertextWrapper>> arg0;
  vector<shared_ptr<SealCiphertextWrapper>> arg1;
  vector<shared_ptr<SealCiphertextWrapper>> out;
  vector<float> exp_out(count);

  for (size_t i = 0; i < count; i++) {
    arg0.emplace_back(make_shared<SealCiphertextWrapper>());
    arg1.emplace_back(make_shared<SealCiphertextWrapper>());
    out.emplace_back(make_shared<SealCiphertextWrapper>());

    arg0[i]->known_value() = true;
    arg0[i]->value() = i;

    arg1[i]->known_value() = true;
    arg1[i]->value() = i * 1.23;

    exp_out[i] = arg0[i]->value() * arg1[i]->value();
  }
  multiply_seal(arg0, arg1, out, ngraph::element::f32, *he_backend, count);

  for (size_t i = 0; i < count; ++i) {
    auto c_out = out[i];
    EXPECT_TRUE(c_out->known_value());
    EXPECT_EQ(c_out->value(), exp_out[i]);
  }
}

NGRAPH_TEST(${BACKEND_NAME}, known_cipher_cipher_mult) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  size_t count = 10;

  vector<shared_ptr<SealCiphertextWrapper>> arg0;
  vector<shared_ptr<SealCiphertextWrapper>> arg1;
  vector<shared_ptr<SealCiphertextWrapper>> out;
  vector<float> exp_out(count);

  for (size_t i = 0; i < count; i++) {
    float arg0_val = i + 1;
    float arg1_val = i * 1.23;
    arg0.emplace_back(make_shared<SealCiphertextWrapper>());
    arg1.emplace_back(make_shared<SealCiphertextWrapper>());
    out.emplace_back(make_shared<SealCiphertextWrapper>());

    he_backend->encrypt(arg0[i], HEPlaintext(arg0_val), ngraph::element::f32,
                        false);
    arg0[i]->known_value() = false;

    arg1[i]->known_value() = true;
    arg1[i]->value() = arg1_val;

    exp_out[i] = arg0_val * arg1_val;
  }
  multiply_seal(arg0, arg1, out, ngraph::element::f32, *he_backend, count);

  for (size_t i = 0; i < count; ++i) {
    EXPECT_EQ(out[i]->known_value(), i == 0);
    HEPlaintext p;
    he_backend->decrypt(p, *out[i]);
    EXPECT_NEAR(p.values()[0], exp_out[i], 1e-3f);
  }
}

NGRAPH_TEST(${BACKEND_NAME}, known_cipher_plain_mult) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  size_t count = 10;

  vector<shared_ptr<SealCiphertextWrapper>> arg0;
  vector<HEPlaintext> arg1;
  vector<shared_ptr<SealCiphertextWrapper>> out;
  vector<float> exp_out(count);

  for (size_t i = 0; i < count; i++) {
    float arg1_val = i * 1.23f;
    arg0.emplace_back(make_shared<SealCiphertextWrapper>());
    arg1.emplace_back(HEPlaintext(arg1_val));
    out.emplace_back(make_shared<SealCiphertextWrapper>());

    arg0[i]->known_value() = true;
    arg0[i]->value() = i;

    exp_out[i] = arg0[i]->value() * arg1_val;
  }
  multiply_seal(arg0, arg1, out, ngraph::element::f32, *he_backend, count);

  for (size_t i = 0; i < count; ++i) {
    auto c_out = out[i];
    EXPECT_TRUE(c_out->known_value());
    EXPECT_EQ(c_out->value(), exp_out[i]);
  }
}

NGRAPH_TEST(${BACKEND_NAME}, known_cipher_cipher_sub) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  size_t count = 10;

  vector<shared_ptr<SealCiphertextWrapper>> arg0;
  vector<shared_ptr<SealCiphertextWrapper>> arg1;
  vector<shared_ptr<SealCiphertextWrapper>> out;
  vector<float> exp_out(count);

  for (size_t i = 0; i < count; i++) {
    float arg0_val = i + 1;
    float arg1_val = i * 1.23;
    arg0.emplace_back(make_shared<SealCiphertextWrapper>());
    arg1.emplace_back(make_shared<SealCiphertextWrapper>());
    out.emplace_back(make_shared<SealCiphertextWrapper>());

    he_backend->encrypt(arg0[i], HEPlaintext(arg0_val), ngraph::element::f32,
                        false);
    arg0[i]->known_value() = false;

    arg1[i]->known_value() = true;
    arg1[i]->value() = arg1_val;

    exp_out[i] = arg0_val - arg1_val;
  }
  subtract_seal(arg0, arg1, out, ngraph::element::f32, *he_backend, count);

  for (size_t i = 0; i < count; ++i) {
    EXPECT_EQ(out[i]->known_value(), false);
    HEPlaintext p;
    he_backend->decrypt(p, *out[i]);
    EXPECT_NEAR(p.values()[0], exp_out[i], 1e-3f);
  }
}

NGRAPH_TEST(${BACKEND_NAME}, known_cipher_plain_sub) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  size_t count = 10;

  vector<shared_ptr<SealCiphertextWrapper>> arg0;
  vector<HEPlaintext> arg1;
  vector<shared_ptr<SealCiphertextWrapper>> out;
  vector<float> exp_out(count);

  for (size_t i = 0; i < count; i++) {
    float arg1_val = i * 1.23f;
    arg0.emplace_back(make_shared<SealCiphertextWrapper>());
    arg1.emplace_back(HEPlaintext(arg1_val));
    out.emplace_back(make_shared<SealCiphertextWrapper>());

    arg0[i]->known_value() = true;
    arg0[i]->value() = i;

    exp_out[i] = arg0[i]->value() - arg1_val;
  }
  subtract_seal(arg0, arg1, out, ngraph::element::f32, *he_backend, count);

  for (size_t i = 0; i < count; ++i) {
    auto c_out = out[i];
    EXPECT_TRUE(c_out->known_value());
    EXPECT_EQ(c_out->value(), exp_out[i]);
  }
}

NGRAPH_TEST(${BACKEND_NAME}, plain_known_cipher_sub) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  size_t count = 10;

  vector<shared_ptr<SealCiphertextWrapper>> arg0;
  vector<HEPlaintext> arg1;
  vector<shared_ptr<SealCiphertextWrapper>> out;
  vector<float> exp_out(count);

  for (size_t i = 0; i < count; i++) {
    float arg1_val = i * 1.23f;
    arg0.emplace_back(make_shared<SealCiphertextWrapper>());
    arg1.emplace_back(HEPlaintext(arg1_val));
    out.emplace_back(make_shared<SealCiphertextWrapper>());

    arg0[i]->known_value() = true;
    arg0[i]->value() = i;

    exp_out[i] = arg1_val - arg0[i]->value();
  }
  subtract_seal(arg1, arg0, out, ngraph::element::f32, *he_backend, count);

  for (size_t i = 0; i < count; ++i) {
    auto c_out = out[i];
    EXPECT_TRUE(c_out->known_value());
    EXPECT_EQ(c_out->value(), exp_out[i]);
  }
}

NGRAPH_TEST(${BACKEND_NAME}, known_cipher_negate) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  size_t count = 10;

  vector<shared_ptr<SealCiphertextWrapper>> arg0;
  vector<shared_ptr<SealCiphertextWrapper>> out;
  vector<float> exp_out(count);

  for (size_t i = 0; i < count; i++) {
    float arg0_val = i + 1;
    arg0.emplace_back(make_shared<SealCiphertextWrapper>());
    out.emplace_back(make_shared<SealCiphertextWrapper>());

    arg0[i]->known_value() = true;
    arg0[i]->value() = arg0_val;

    exp_out[i] = -arg0_val;
  }
  negate_seal(arg0, out, ngraph::element::f32, *he_backend, count);

  for (size_t i = 0; i < count; ++i) {
    EXPECT_TRUE(out[i]->known_value());
    HEPlaintext p;
    he_backend->decrypt(p, *out[i]);
    EXPECT_NEAR(p.values()[0], exp_out[i], 1e-3f);
  }
}

NGRAPH_TEST(${BACKEND_NAME}, known_cipher_relu) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  int count = 10;

  vector<shared_ptr<SealCiphertextWrapper>> arg0;
  vector<shared_ptr<SealCiphertextWrapper>> out;
  vector<float> exp_out(count);

  auto relu = [](float f) { return f > 0 ? f : 0.f; };

  for (int i = 0; i < count; i++) {
    float arg0_val = i - count / 2;
    arg0.emplace_back(make_shared<SealCiphertextWrapper>());
    out.emplace_back(make_shared<SealCiphertextWrapper>());

    arg0[i]->known_value() = true;
    arg0[i]->value() = arg0_val;

    exp_out[i] = relu(arg0_val);
  }
  relu_seal(arg0, out, count, *he_backend);

  for (int i = 0; i < count; ++i) {
    EXPECT_TRUE(out[i]->known_value());
    HEPlaintext p;
    he_backend->decrypt(p, *out[i]);
    EXPECT_NEAR(p.values()[0], exp_out[i], 1e-3f);
  }
}

NGRAPH_TEST(${BACKEND_NAME}, known_cipher_bounded_relu) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  int count = 10;

  vector<shared_ptr<SealCiphertextWrapper>> arg0;
  vector<shared_ptr<SealCiphertextWrapper>> out;
  vector<float> exp_out(count);

  float alpha = 6.0f;
  auto bounded_relu = [alpha](double f) {
    return f > alpha ? alpha : (f > 0) ? f : 0.f;
  };

  for (int i = 0; i < count; i++) {
    float arg0_val = i - count / 2;
    arg0.emplace_back(make_shared<SealCiphertextWrapper>());
    out.emplace_back(make_shared<SealCiphertextWrapper>());

    arg0[i]->known_value() = true;
    arg0[i]->value() = arg0_val;

    exp_out[i] = bounded_relu(arg0_val);
  }
  bounded_relu_seal(arg0, out, count, alpha, *he_backend);

  for (int i = 0; i < count; ++i) {
    EXPECT_TRUE(out[i]->known_value());
    HEPlaintext p;
    he_backend->decrypt(p, *out[i]);
    EXPECT_NEAR(p.values()[0], exp_out[i], 1e-3f);
  }
}
