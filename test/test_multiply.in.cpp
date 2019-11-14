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
#include "seal/kernel/multiply_seal.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

static std::string s_manifest = "${MANIFEST}";

namespace ngraph::he {

auto mult_test = [](const Shape& shape, const bool arg1_encrypted,
                    const bool arg2_encrypted, const bool complex_packing,
                    const bool packed) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        HESealEncryptionParameters::default_complex_packing_parms());
  }

  auto a = std::make_shared<op::Parameter>(element::f32, shape);
  auto b = std::make_shared<op::Parameter>(element::f32, shape);
  auto t = std::make_shared<op::Multiply>(a, b);
  auto f = std::make_shared<Function>(t, ParameterVector{a, b});

  auto t_a =
      test::tensor_from_flags(*he_backend, shape, arg1_encrypted, packed);
  auto t_b =
      test::tensor_from_flags(*he_backend, shape, arg2_encrypted, packed);
  auto t_result = test::tensor_from_flags(
      *he_backend, shape, arg1_encrypted || arg2_encrypted, packed);

  const auto& arg1_config =
      test::config_from_flags(false, arg1_encrypted, packed);
  const auto& arg2_config =
      test::config_from_flags(false, arg2_encrypted, packed);

  std::string error_str;
  he_backend->set_config(
      {{a->get_name(), arg1_config}, {b->get_name(), arg2_config}}, error_str);

  std::vector<float> input_a;
  std::vector<float> input_b;
  std::vector<float> exp_result;

  for (int i = 0; i < shape_size(shape); ++i) {
    input_a.emplace_back(i);

    if (i % 2 == 0) {
      input_b.emplace_back(i);
    } else {
      input_b.emplace_back(1 - i);
    }

    if (packed && complex_packing && arg1_encrypted && !arg2_encrypted) {
      exp_result.emplace_back(
          input_a.back() *
          input_b[i % shape_size(HETensor::pack_shape(shape))]);
    } else if (packed && complex_packing && !arg1_encrypted && arg2_encrypted) {
      exp_result.emplace_back(
          input_a[i % shape_size(HETensor::pack_shape(shape))] *
          input_b.back());
    } else {
      exp_result.emplace_back(input_a.back() * input_b.back());
    }
  }
  copy_data(t_a, input_a);
  copy_data(t_b, input_b);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a, t_b});
  EXPECT_TRUE(test::all_close(read_vector<float>(t_result), exp_result, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, mult_2_3_plain_plain_real_unpacked) {
  mult_test(Shape{2, 3}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, mult_2_3_plain_plain_real_packed) {
  mult_test(Shape{2, 3}, false, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, mult_2_3_plain_plain_complex_unpacked) {
  mult_test(Shape{2, 3}, false, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, mult_2_3_plain_plain_complex_packed) {
  mult_test(Shape{2, 3}, false, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, mult_2_3_plain_cipher_real_unpacked) {
  mult_test(Shape{2, 3}, false, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, mult_2_3_plain_cipher_real_packed) {
  mult_test(Shape{2, 3}, false, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, mult_2_3_plain_cipher_complex_unpacked) {
  mult_test(Shape{2, 3}, false, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, mult_2_3_plain_cipher_complex_packed) {
  mult_test(Shape{2, 3}, false, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, mult_2_3_cipher_plain_real_unpacked) {
  mult_test(Shape{2, 3}, true, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, mult_2_3_cipher_plain_real_packed) {
  mult_test(Shape{2, 3}, true, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, mult_2_3_cipher_plain_complex_unpacked) {
  mult_test(Shape{2, 3}, true, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, mult_2_3_cipher_plain_complex_packed) {
  mult_test(Shape{2, 3}, true, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, mult_2_3_cipher_cipher_real_unpacked) {
  mult_test(Shape{2, 3}, true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, mult_2_3_cipher_cipher_real_packed) {
  mult_test(Shape{2, 3}, true, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, mult_2_3_cipher_cipher_complex_unpacked) {
  mult_test(Shape{2, 3}, true, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, mult_2_3_cipher_cipher_complex_packed) {
  mult_test(Shape{2, 3}, true, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, mult_end_of_depth) {
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

  Shape shape{2, 2};

  bool arg1_encrypted = true;
  bool arg2_encrypted = true;
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

  std::vector<float> input_a{1, 2, 3, 4};
  std::vector<float> input_b{5, 6, 7, 8};
  std::vector<float> exp_result{5, 12, 21, 32};

  copy_data(t_a, input_a);
  copy_data(t_b, input_b);

  auto handle = backend->compile(f);
  EXPECT_ANY_THROW(handle->call_with_validate({t_result}, {t_a, t_b}));
}

NGRAPH_TEST(${BACKEND_NAME}, mult_zero) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{2, 2};

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

  std::vector<float> input_a{0, 0, 3, 4};
  std::vector<float> input_b{5, 6, 0, 0};
  std::vector<float> exp_result{0, 0, 0, 0};

  copy_data(t_a, input_a);
  copy_data(t_b, input_b);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a, t_b});
  EXPECT_TRUE(test::all_close(read_vector<float>(t_result), exp_result, 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, mult_wrong_output_type) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{1};

  bool packed = false;

  std::vector<float> cipher_input{7};
  std::vector<float> plain_input{8};

  // cipher cipher => plain
  {
    auto cipher_tensor = std::static_pointer_cast<HETensor>(
        test::tensor_from_flags(*he_backend, shape, true, packed));
    auto plain_tensor = std::static_pointer_cast<HETensor>(
        test::tensor_from_flags(*he_backend, shape, false, packed));
    auto result_tensor = std::static_pointer_cast<HETensor>(
        test::tensor_from_flags(*he_backend, shape, false, packed));

    copy_data(cipher_tensor, cipher_input);
    copy_data(plain_tensor, plain_input);

    scalar_multiply_seal(cipher_tensor->data(0), cipher_tensor->data(0),
                         result_tensor->data(0), *he_backend);
    EXPECT_TRUE(test::all_close(read_vector<float>(result_tensor),
                                std::vector<float>{49}, 1e-3f));
  }
  // cipher plain => plain
  {
    auto cipher_tensor = std::static_pointer_cast<HETensor>(
        test::tensor_from_flags(*he_backend, shape, true, packed));
    auto plain_tensor = std::static_pointer_cast<HETensor>(
        test::tensor_from_flags(*he_backend, shape, false, packed));
    auto result_tensor = std::static_pointer_cast<HETensor>(
        test::tensor_from_flags(*he_backend, shape, false, packed));

    copy_data(cipher_tensor, cipher_input);
    copy_data(plain_tensor, plain_input);

    scalar_multiply_seal(cipher_tensor->data(0), plain_tensor->data(0),
                         result_tensor->data(0), *he_backend);
    EXPECT_TRUE(test::all_close(read_vector<float>(result_tensor),
                                std::vector<float>{56}, 1e-3f));
  }
  // plain cipher => plain
  {
    auto cipher_tensor = std::static_pointer_cast<HETensor>(
        test::tensor_from_flags(*he_backend, shape, true, packed));
    auto plain_tensor = std::static_pointer_cast<HETensor>(
        test::tensor_from_flags(*he_backend, shape, false, packed));
    auto result_tensor = std::static_pointer_cast<HETensor>(
        test::tensor_from_flags(*he_backend, shape, false, packed));

    copy_data(cipher_tensor, cipher_input);
    copy_data(plain_tensor, plain_input);

    scalar_multiply_seal(cipher_tensor->data(0), plain_tensor->data(0),
                         result_tensor->data(0), *he_backend);
    EXPECT_TRUE(test::all_close(read_vector<float>(result_tensor),
                                std::vector<float>{56}, 1e-3f));
  }
  // plain plain => cipher
  {
    auto plain_tensor = std::static_pointer_cast<HETensor>(
        test::tensor_from_flags(*he_backend, shape, false, packed));
    auto result_tensor = std::static_pointer_cast<HETensor>(
        test::tensor_from_flags(*he_backend, shape, true, packed));

    copy_data(plain_tensor, plain_input);

    scalar_multiply_seal(plain_tensor->data(0), plain_tensor->data(0),
                         result_tensor->data(0), *he_backend);
    EXPECT_TRUE(test::all_close(read_vector<float>(result_tensor),
                                std::vector<float>{64}, 1e-3f));
  }
}

}  // namespace ngraph::he