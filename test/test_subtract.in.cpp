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
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::he;

static string s_manifest = "${MANIFEST}";

auto sub_test = [](const Shape& shape, const bool arg1_encrypted,
                   const bool arg2_encrypted, const bool complex_packing,
                   const bool packed) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        HESealEncryptionParameters::default_complex_packing_parms());
  }

  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Subtract>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  a->set_op_annotations(
      test::he::annotation_from_flags(false, arg1_encrypted, packed));
  b->set_op_annotations(
      test::he::annotation_from_flags(false, arg2_encrypted, packed));

  auto t_a =
      test::he::tensor_from_flags(*he_backend, shape, arg1_encrypted, packed);
  auto t_b =
      test::he::tensor_from_flags(*he_backend, shape, arg2_encrypted, packed);
  auto t_result = test::he::tensor_from_flags(
      *he_backend, shape, arg1_encrypted || arg2_encrypted, packed);

  vector<float> input_a;
  vector<float> input_b;
  vector<float> exp_result;

  for (int i = 0; i < shape_size(shape); ++i) {
    input_a.emplace_back(i);

    if (i % 2 == 0) {
      input_b.emplace_back(i);
    } else {
      input_b.emplace_back(1 - i);
    }
    exp_result.emplace_back(input_a.back() - input_b.back());
  }
  copy_data(t_a, input_a);
  copy_data(t_b, input_b);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a, t_b});

  EXPECT_TRUE(
      test::he::all_close(read_vector<float>(t_result), exp_result, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_plain_plain_real_unpacked) {
  sub_test(Shape{2, 3}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_plain_plain_real_packed) {
  sub_test(Shape{2, 3}, false, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_plain_plain_complex_unpacked) {
  sub_test(Shape{2, 3}, false, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_plain_plain_complex_packed) {
  sub_test(Shape{2, 3}, false, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_plain_cipher_real_unpacked) {
  sub_test(Shape{2, 3}, false, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_plain_cipher_real_packed) {
  sub_test(Shape{2, 3}, false, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_plain_cipher_complex_unpacked) {
  sub_test(Shape{2, 3}, false, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_plain_cipher_complex_packed) {
  sub_test(Shape{2, 3}, false, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_cipher_plain_real_unpacked) {
  sub_test(Shape{2, 3}, true, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_cipher_plain_real_packed) {
  sub_test(Shape{2, 3}, true, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_cipher_plain_complex_unpacked) {
  sub_test(Shape{2, 3}, true, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_cipher_plain_complex_packed) {
  sub_test(Shape{2, 3}, true, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_cipher_cipher_real_unpacked) {
  sub_test(Shape{2, 3}, true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_cipher_cipher_real_packed) {
  sub_test(Shape{2, 3}, true, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_cipher_cipher_complex_unpacked) {
  sub_test(Shape{2, 3}, true, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_cipher_cipher_complex_packed) {
  sub_test(Shape{2, 3}, true, true, true, true);
}
