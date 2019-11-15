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

namespace ngraph::runtime::he {

auto constant_test = [](const bool arg1_encrypted, const bool arg2_encrypted) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<he::HESealBackend*>(backend.get());

  Shape shape{2, 2};
  auto a = op::Constant::create(element::f32, shape, {1, 2, 3, 4});
  auto b = std::make_shared<op::Parameter>(element::f32, shape);
  auto c = std::make_shared<op::Parameter>(element::f32, shape);
  auto t = (a + b) * c;
  auto f = std::make_shared<Function>(t, ParameterVector{b, c});

  const auto& arg1_config =
      test::config_from_flags(false, arg1_encrypted, false);
  const auto& arg2_config =
      test::config_from_flags(false, arg2_encrypted, false);

  std::string error_str;
  he_backend->set_config(
      {{b->get_name(), arg1_config}, {c->get_name(), arg2_config}}, error_str);

  auto t_b = test::tensor_from_flags(*he_backend, shape, arg1_encrypted, false);
  auto t_c = test::tensor_from_flags(*he_backend, shape, arg2_encrypted, false);
  auto t_result = test::tensor_from_flags(
      *he_backend, shape, arg1_encrypted || arg2_encrypted, false);

  std::vector<float> input_b{5, 6, 7, 8};
  std::vector<float> input_c{9, 10, 11, 12};
  std::vector<float> exp_result{54, 80, 110, 144};

  copy_data(t_b, input_b);
  copy_data(t_c, input_c);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_b, t_c});
  EXPECT_TRUE(test::all_close(read_vector<float>(t_result), exp_result, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, constant) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape{2, 2};
  {
    auto a = op::Constant::create(element::f32, shape, {0.1, 0.2, 0.3, 0.4});
    auto f = std::make_shared<Function>(a, ParameterVector{});
    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {});
    EXPECT_TRUE(test::all_close((std::vector<float>{0.1, 0.2, 0.3, 0.4}),
                                read_vector<float>(result)));
  }
  {
    auto a = op::Constant::create(element::f64, shape, {0.1, 0.2, 0.3, 0.4});
    auto f = std::make_shared<Function>(a, ParameterVector{});
    auto result = backend->create_tensor(element::f64, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {});
    EXPECT_TRUE(test::all_close((std::vector<double>{0.1, 0.2, 0.3, 0.4}),
                                read_vector<double>(result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, constant_abc_plain_plain) {
  constant_test(false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, constant_abc_plain_cipher) {
  constant_test(false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, constant_abc_cipher_plain) {
  constant_test(true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, constant_abc_cipher_cipher) {
  constant_test(true, true);
}

}  // namespace ngraph::runtime::he
