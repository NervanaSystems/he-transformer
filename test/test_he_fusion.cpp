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

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "op/bounded_relu.hpp"
#include "pass/he_fusion.hpp"
#include "seal/he_seal_backend.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

namespace ngraph::he {

static void check_bounded_relu(const Shape& param_shape, float constant_val) {
  auto make_function = [](Shape input_shape, float alpha_val) {
    auto relu_input =
        std::make_shared<op::Parameter>(element::f32, input_shape);
    auto relu = std::make_shared<op::Relu>(relu_input);
    auto alpha = op::Constant::create<float>(
        element::f32, input_shape, std::vector<float>(1.0f, alpha_val));
    auto min = std::make_shared<op::Minimum>(relu, alpha);
    auto f = std::make_shared<Function>(NodeVector{min},
                                        ParameterVector{relu_input});
    return f;
  };

  auto he_f = make_function(param_shape, constant_val);
  auto int_f = make_function(param_shape, constant_val);
  ngraph::test::Uniform<float> rng(-10.0f, 10.0f);
  std::vector<std::vector<float>> args;

  for (const auto& param : int_f->get_parameters()) {
    std::vector<float> tensor_val(shape_size(param->get_shape()));
    rng.initialize(tensor_val);
    args.push_back(tensor_val);
  }

  auto he_backend_orig = runtime::Backend::create("HE_SEAL");
  auto he_backend = static_cast<HESealBackend*>(he_backend_orig.get());
  auto he_handle = he_backend->compile(he_f);
  EXPECT_EQ(1, count_ops_of_type<op::BoundedRelu>(he_f));

  auto he_a = he_backend->create_plain_tensor(element::f32, param_shape);
  auto he_result = he_backend->create_plain_tensor(element::f32, param_shape);
  copy_data(he_a, args[0]);
  he_handle->call_with_validate({he_result}, {he_a});

  auto int_backend = runtime::Backend::create("INTERPRETER");
  auto int_handle = int_backend->compile(int_f);
  auto int_a = int_backend->create_tensor(element::f32, param_shape);
  auto int_result = int_backend->create_tensor(element::f32, param_shape);
  copy_data(int_a, args[0]);
  int_handle->call_with_validate({int_result}, {int_a});

  EXPECT_TRUE(test::all_close(read_vector<float>(he_result),
                              read_vector<float>(int_result), 1e-3f));
}

TEST(he_fusion, bounded_relu_fusion) {
  check_bounded_relu(Shape{4, 3, 2, 2}, 8.0f);
  check_bounded_relu(Shape{4, 3}, 4.0f);
  check_bounded_relu(Shape{4, 3, 2}, 2.0f);
}

TEST(he_fusion, bounded_relu_no_fusion) {
  auto check_no_fusion = [](std::shared_ptr<Function> function) {
    auto backend = runtime::Backend::create("HE_SEAL");
    auto he_backend = static_cast<HESealBackend*>(backend.get());
    auto handle = he_backend->compile(function);
    EXPECT_EQ(uint64_t{0}, count_ops_of_type<op::BoundedRelu>(function));
  };

  // Wrong type
  {
    Shape shape{2, 2};
    float alpha_val{7.2};
    auto relu_input = std::make_shared<op::Parameter>(element::f64, shape);
    auto relu = std::make_shared<op::Relu>(relu_input);
    auto alpha = op::Constant::create<double>(
        element::f64, shape, std::vector<double>(1.0, alpha_val));
    auto min = std::make_shared<op::Minimum>(relu, alpha);
    auto f = std::make_shared<Function>(NodeVector{min},
                                        ParameterVector{relu_input});

    check_no_fusion(f);
  }
  /// Alpha is not constant
  {
    Shape shape{2, 2};
    auto relu_input = std::make_shared<op::Parameter>(element::f32, shape);
    auto relu = std::make_shared<op::Relu>(relu_input);
    auto alpha = std::make_shared<op::Parameter>(element::f32, shape);
    auto min = std::make_shared<op::Minimum>(relu, alpha);
    auto f = std::make_shared<Function>(NodeVector{min},
                                        ParameterVector{relu_input, alpha});

    check_no_fusion(f);
  }
}
}  // namespace ngraph::he
