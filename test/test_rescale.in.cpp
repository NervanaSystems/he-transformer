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

NGRAPH_TEST(${BACKEND_NAME}, skip_rescale_lowest) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  std::string param_str = R"(
    {
        "scheme_name" : "HE_SEAL",
        "poly_modulus_degree" : 2048,
        "security_level" : 128,
        "coeff_modulus" : [54],
        "scale" : 16777216,
        "complex_packing" : true
    })";
  auto he_parms = HESealEncryptionParameters::parse_config_or_use_default(
      param_str.c_str());
  he_backend->update_encryption_parameters(he_parms);

  Shape shape{3, 1};

  bool arg1_encrypted = true;
  bool arg2_encrypted = false;
  bool packed = false;

  auto a = std::make_shared<op::Parameter>(element::f32, shape);
  auto b = std::make_shared<op::Parameter>(element::f32, shape);
  auto t = std::make_shared<op::Multiply>(a, b);
  auto f = std::make_shared<Function>(t, ParameterVector{a, b});

  const auto& arg1_config =
      test::config_from_flags(false, arg1_encrypted, packed);
  const auto& arg2_config =
      test::config_from_flags(false, arg2_encrypted, packed);

  std::string error_str;
  he_backend->set_config(
      {{a->get_name(), arg1_config}, {b->get_name(), arg2_config}}, error_str);

  auto t_a =
      test::tensor_from_flags(*he_backend, shape, arg1_encrypted, packed);
  auto t_b =
      test::tensor_from_flags(*he_backend, shape, arg2_encrypted, packed);
  auto t_result = test::tensor_from_flags(
      *he_backend, shape, arg1_encrypted || arg2_encrypted, packed);

  std::vector<float> input_a{1, 2, 3};
  std::vector<float> input_b{5, 6, 7};
  std::vector<float> exp_result{5, 12, 21};

  copy_data(t_a, input_a);
  copy_data(t_b, input_b);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a, t_b});
  EXPECT_TRUE(test::all_close(read_vector<float>(t_result), exp_result, 1e-1f));
}

}  // namespace ngraph::runtime::he
