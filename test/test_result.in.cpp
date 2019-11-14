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

namespace ngraph::he {

auto result_test = [](const bool input_encrypted, const bool output_encrypted) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{2, 3};

  auto a = std::make_shared<op::Parameter>(element::f32, shape);
  auto t = std::make_shared<op::Relu>(a);
  auto f = std::make_shared<Function>(t, ParameterVector{a});

  const auto& config =
      test::config_from_flags(false, input_encrypted, false);

  std::string error_str;
  he_backend->set_config({{a->get_name(), config}}, error_str);

  auto t_a =
      test::tensor_from_flags(*he_backend, shape, input_encrypted, false);
  auto t_result =
      test::tensor_from_flags(*he_backend, shape, output_encrypted, false);

  std::vector<float> input_a{-2, -1, 0, 1, 2, 3};
  std::vector<float> exp_result{0, 0, 0, 1, 2, 3};
  copy_data(t_a, input_a);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a});
  EXPECT_TRUE(
      test::all_close(read_vector<float>(t_result), exp_result, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, result_cipher_to_cipher) {
  result_test(true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, result_cipher_to_plain) {
  result_test(true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, result_plain_to_cipher) {
  result_test(true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, result_plain_to_plain) {
  result_test(true, false);
}

}  // namespace ngraph::he
