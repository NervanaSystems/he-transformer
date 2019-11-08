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

static std::string s_manifest = "${MANIFEST}";

auto exp_test = [](const ngraph::Shape& shape, const bool arg1_encrypted,
                   const bool complex_packing, const bool packed) {
  auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        ngraph::he::HESealEncryptionParameters::
            default_complex_packing_parms());
  }

  auto a = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto t = std::make_shared<ngraph::op::Exp>(a);
  auto f = std::make_shared<ngraph::Function>(t, ngraph::ParameterVector{a});
  a->set_op_annotations(
      ngraph::test::he::annotation_from_flags(false, arg1_encrypted, packed));

  auto t_a = ngraph::test::he::tensor_from_flags(*he_backend, shape,
                                                 arg1_encrypted, packed);
  auto t_result = ngraph::test::he::tensor_from_flags(*he_backend, shape,
                                                      arg1_encrypted, packed);

  std::vector<float> input_a;
  std::vector<float> exp_result;

  for (int i = 0; i < shape_size(shape); ++i) {
    if (i % 2 == 0) {
      input_a.emplace_back(i);
    } else {
      input_a.emplace_back(1 - i);
    }
    exp_result.emplace_back(std::exp(input_a.back()));
  }
  copy_data(t_a, input_a);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a});
  EXPECT_TRUE(ngraph::test::he::all_close(read_vector<float>(t_result),
                                          exp_result, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, exp_2_3_plain_real_unpacked) {
  exp_test(ngraph::Shape{2, 3}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, exp_2_3_plain_real_packed) {
  exp_test(ngraph::Shape{2, 3}, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, exp_2_3_plain_complex_unpacked) {
  exp_test(ngraph::Shape{2, 3}, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, exp_2_3_plain_complex_packed) {
  exp_test(ngraph::Shape{2, 3}, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, exp_2_3_cipher_real_unpacked) {
  exp_test(ngraph::Shape{2, 3}, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, exp_2_3_cipher_real_packed) {
  exp_test(ngraph::Shape{2, 3}, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, exp_2_3_cipher_complex_unpacked) {
  exp_test(ngraph::Shape{2, 3}, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, exp_2_3_cipher_complex_packed) {
  exp_test(ngraph::Shape{2, 3}, true, true, true);
}
