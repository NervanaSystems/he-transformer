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

auto softmax_test = [](const Shape& shape_a, const AxisSet& axes,
                       const vector<float>& input_a,
                       const vector<float>& output, const bool arg1_encrypted,
                       const bool complex_packing, const bool packed) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        HESealEncryptionParameters::default_complex_packing_parms());
  }

  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto t = make_shared<op::Softmax>(a, axes);
  auto f = make_shared<Function>(t, ParameterVector{a});

  a->set_op_annotations(
      test::he::annotation_from_flags(false, arg1_encrypted, packed));

  auto t_a =
      test::he::tensor_from_flags(*he_backend, shape_a, arg1_encrypted, packed);
  auto t_result = test::he::tensor_from_flags(*he_backend, t->get_shape(),
                                              arg1_encrypted, packed);

  copy_data(t_a, input_a);

  auto handle = backend->compile(f);
  if (packed && (axes.find(0) != axes.end())) {
    EXPECT_ANY_THROW((handle->call_with_validate({t_result}, {t_a})));
  } else {
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_TRUE(
        test::he::all_close(read_vector<float>(t_result), output, 1e-3f));
  }
};

NGRAPH_TEST(${BACKEND_NAME}, softmax_all_plain_real_unpacked) {
  auto d = expf(-3) + expf(-2) + expf(-1) + expf(0) + expf(1) + expf(2);
  softmax_test(Shape{2, 3}, AxisSet{0, 1}, vector<float>{-3, -2, -1, 0, 1, 2},
               vector<float>{expf(-3) / d, expf(-2) / d, expf(-1) / d,
                             expf(0) / d, expf(1) / d, expf(2) / d},
               false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_all_plain_real_packed) {
  auto d = expf(-3) + expf(-2) + expf(-1) + expf(0) + expf(1) + expf(2);
  softmax_test(Shape{2, 3}, AxisSet{0, 1}, vector<float>{-3, -2, -1, 0, 1, 2},
               vector<float>{expf(-3) / d, expf(-2) / d, expf(-1) / d,
                             expf(0) / d, expf(1) / d, expf(2) / d},
               false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_all_plain_complex_unpacked) {
  auto d = expf(-3) + expf(-2) + expf(-1) + expf(0) + expf(1) + expf(2);
  softmax_test(Shape{2, 3}, AxisSet{0, 1}, vector<float>{-3, -2, -1, 0, 1, 2},
               vector<float>{expf(-3) / d, expf(-2) / d, expf(-1) / d,
                             expf(0) / d, expf(1) / d, expf(2) / d},
               false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_all_plain_complex_packed) {
  auto d = expf(-3) + expf(-2) + expf(-1) + expf(0) + expf(1) + expf(2);
  softmax_test(Shape{2, 3}, AxisSet{0, 1}, vector<float>{-3, -2, -1, 0, 1, 2},
               vector<float>{expf(-3) / d, expf(-2) / d, expf(-1) / d,
                             expf(0) / d, expf(1) / d, expf(2) / d},
               false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_all_cipher_real_unpacked) {
  auto d = expf(-3) + expf(-2) + expf(-1) + expf(0) + expf(1) + expf(2);
  softmax_test(Shape{2, 3}, AxisSet{0, 1}, vector<float>{-3, -2, -1, 0, 1, 2},
               vector<float>{expf(-3) / d, expf(-2) / d, expf(-1) / d,
                             expf(0) / d, expf(1) / d, expf(2) / d},
               true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_all_cipher_real_packed) {
  auto d = expf(-3) + expf(-2) + expf(-1) + expf(0) + expf(1) + expf(2);
  softmax_test(Shape{2, 3}, AxisSet{0, 1}, vector<float>{-3, -2, -1, 0, 1, 2},
               vector<float>{expf(-3) / d, expf(-2) / d, expf(-1) / d,
                             expf(0) / d, expf(1) / d, expf(2) / d},
               true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_all_cipher_complex_unpacked) {
  auto d = expf(-3) + expf(-2) + expf(-1) + expf(0) + expf(1) + expf(2);
  softmax_test(Shape{2, 3}, AxisSet{0, 1}, vector<float>{-3, -2, -1, 0, 1, 2},
               vector<float>{expf(-3) / d, expf(-2) / d, expf(-1) / d,
                             expf(0) / d, expf(1) / d, expf(2) / d},
               true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_all_cipher_complex_packed) {
  auto d = expf(-3) + expf(-2) + expf(-1) + expf(0) + expf(1) + expf(2);
  softmax_test(Shape{2, 3}, AxisSet{0, 1}, vector<float>{-3, -2, -1, 0, 1, 2},
               vector<float>{expf(-3) / d, expf(-2) / d, expf(-1) / d,
                             expf(0) / d, expf(1) / d, expf(2) / d},
               true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_plain_real_unpacked) {
  auto d0 = expf(-10) + expf(-20) + expf(-30);
  auto d1 = expf(-40) + expf(-50) + expf(-60);
  softmax_test(Shape{2, 3}, AxisSet{1},
               vector<float>{-10, -20, -30, -40, -50, -60},
               vector<float>{expf(-10) / d0, expf(-20) / d0, expf(-30) / d0,
                             expf(-40) / d1, expf(-50) / d1, expf(-60) / d1},
               false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_plain_real_packed) {
  auto d0 = expf(-10) + expf(-20) + expf(-30);
  auto d1 = expf(-40) + expf(-50) + expf(-60);
  softmax_test(Shape{2, 3}, AxisSet{1},
               vector<float>{-10, -20, -30, -40, -50, -60},
               vector<float>{expf(-10) / d0, expf(-20) / d0, expf(-30) / d0,
                             expf(-40) / d1, expf(-50) / d1, expf(-60) / d1},
               false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_plain_complex_unpacked) {
  auto d0 = expf(-10) + expf(-20) + expf(-30);
  auto d1 = expf(-40) + expf(-50) + expf(-60);
  softmax_test(Shape{2, 3}, AxisSet{1},
               vector<float>{-10, -20, -30, -40, -50, -60},
               vector<float>{expf(-10) / d0, expf(-20) / d0, expf(-30) / d0,
                             expf(-40) / d1, expf(-50) / d1, expf(-60) / d1},
               false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_plain_complex_packed) {
  auto d0 = expf(-10) + expf(-20) + expf(-30);
  auto d1 = expf(-40) + expf(-50) + expf(-60);
  softmax_test(Shape{2, 3}, AxisSet{1},
               vector<float>{-10, -20, -30, -40, -50, -60},
               vector<float>{expf(-10) / d0, expf(-20) / d0, expf(-30) / d0,
                             expf(-40) / d1, expf(-50) / d1, expf(-60) / d1},
               false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_cipher_real_unpacked) {
  auto d0 = expf(-10) + expf(-20) + expf(-30);
  auto d1 = expf(-40) + expf(-50) + expf(-60);
  softmax_test(Shape{2, 3}, AxisSet{1},
               vector<float>{-10, -20, -30, -40, -50, -60},
               vector<float>{expf(-10) / d0, expf(-20) / d0, expf(-30) / d0,
                             expf(-40) / d1, expf(-50) / d1, expf(-60) / d1},
               true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_cipher_real_packed) {
  auto d0 = expf(-10) + expf(-20) + expf(-30);
  auto d1 = expf(-40) + expf(-50) + expf(-60);
  softmax_test(Shape{2, 3}, AxisSet{1},
               vector<float>{-10, -20, -30, -40, -50, -60},
               vector<float>{expf(-10) / d0, expf(-20) / d0, expf(-30) / d0,
                             expf(-40) / d1, expf(-50) / d1, expf(-60) / d1},
               true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_cipher_complex_unpacked) {
  auto d0 = expf(-10) + expf(-20) + expf(-30);
  auto d1 = expf(-40) + expf(-50) + expf(-60);
  softmax_test(Shape{2, 3}, AxisSet{1},
               vector<float>{-10, -20, -30, -40, -50, -60},
               vector<float>{expf(-10) / d0, expf(-20) / d0, expf(-30) / d0,
                             expf(-40) / d1, expf(-50) / d1, expf(-60) / d1},
               true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_cipher_complex_packed) {
  auto d0 = expf(-10) + expf(-20) + expf(-30);
  auto d1 = expf(-40) + expf(-50) + expf(-60);
  softmax_test(Shape{2, 3}, AxisSet{1},
               vector<float>{-10, -20, -30, -40, -50, -60},
               vector<float>{expf(-10) / d0, expf(-20) / d0, expf(-30) / d0,
                             expf(-40) / d1, expf(-50) / d1, expf(-60) / d1},
               true, true, true);
}
