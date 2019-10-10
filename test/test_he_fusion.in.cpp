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

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

static void check_bounded_relu(Shape param_shape, float constant_val) {
  auto make_function = [](Shape input_shape, float alpha_val) {
    auto relu_input =
        std::make_shared<op::Parameter>(element::f32, input_shape);
    auto relu = std::make_shared<op::Relu>(relu_input);
    auto alpha = op::Constant::create<float>(
        element::f32, input_shape, std::vector<float>(1.0f, alpha_val));
    auto min = std::make_shared<op::Minimum>(relu, alpha);
    auto f =
        make_shared<Function>(NodeVector{min}, ParameterVector{relu_input});
    return f;
  };

  auto he_f = make_function(param_shape, constant_val);
  auto int_f = make_function(param_shape, constant_val);
  test::Uniform<float> rng(-10.0f, 10.0f);
  vector<vector<float>> args;

  for (shared_ptr<op::Parameter> param : int_f->get_parameters()) {
    vector<float> tensor_val(shape_size(param->get_shape()));
    rng.initialize(tensor_val);
    args.push_back(tensor_val);
  }

  auto he_backend_orig = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend =
      static_cast<ngraph::he::HESealBackend*>(he_backend_orig.get());
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

  EXPECT_TRUE(all_close(read_vector<float>(he_result),
                        read_vector<float>(int_result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, bounded_relu_fusion) {
  check_bounded_relu(Shape{4, 3, 2, 2}, 8.0f);
  check_bounded_relu(Shape{4, 3}, 4.0f);
  check_bounded_relu(Shape{4, 3, 2}, 2.0f);
}

NGRAPH_TEST(${BACKEND_NAME}, const_broadcast_add) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape input_shape{};
  Shape output_shape{336};

  std::vector<float> variance(336, 1.23);

  auto eps = op::Constant::create(element::f32, input_shape, {0.01});
  auto var = op::Constant::create(element::f32, output_shape, variance);

  auto bcast = make_shared<op::Broadcast>(eps, output_shape, AxisSet{0});

  auto add = make_shared<op::Add>(bcast, var);

  auto f = make_shared<Function>(NodeVector{add}, ParameterVector{});

  NGRAPH_INFO << "Nodes before compiling";
  for (const auto& node : f->get_ordered_ops()) {
    NGRAPH_INFO << "node " << node->get_name();
  }

  // Includes result op
  EXPECT_EQ(f->get_ordered_ops().size(), 5);

  auto handle = backend->compile(f);

  NGRAPH_INFO << "Nodes after compiling";
  for (const auto& node : f->get_ordered_ops()) {
    NGRAPH_INFO << "node " << node->get_name();
  }

  // Constant plus result op
  EXPECT_EQ(f->get_ordered_ops().size(), 2);
}

NGRAPH_TEST(${BACKEND_NAME}, const_broadcast_add_3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape input_shape{};
  Shape output_shape1{123};
  Shape output_shape2{234};
  Shape output_shape3{456};

  std::vector<float> variance(336, 1.23);

  auto eps = op::Constant::create(element::f32, input_shape, {0.01});
  auto var1 = op::Constant::create(element::f32, output_shape1,
                                   std::vector<float>(123, 1.23));
  auto var2 = op::Constant::create(element::f32, output_shape2,
                                   std::vector<float>(234, 2.34));
  auto var3 = op::Constant::create(element::f32, output_shape3,
                                   std::vector<float>(456, 3.45));

  auto bcast1 = make_shared<op::Broadcast>(eps, output_shape1, AxisSet{0});
  auto bcast2 = make_shared<op::Broadcast>(eps, output_shape2, AxisSet{0});
  auto bcast3 = make_shared<op::Broadcast>(eps, output_shape3, AxisSet{0});

  auto add1 = make_shared<op::Add>(bcast1, var1);
  auto add2 = make_shared<op::Add>(bcast2, var2);
  auto add3 = make_shared<op::Add>(bcast3, var3);

  auto f =
      make_shared<Function>(NodeVector{add1, add2, add3}, ParameterVector{});

  NGRAPH_INFO << "Nodes before compiling";
  for (const auto& node : f->get_ordered_ops()) {
    NGRAPH_INFO << "node " << node->get_name();
  }

  // Includes result op
  EXPECT_EQ(f->get_ordered_ops().size(), 5);

  auto handle = backend->compile(f);

  NGRAPH_INFO << "Nodes after compiling";
  for (const auto& node : f->get_ordered_ops()) {
    NGRAPH_INFO << "node " << node->get_name();
  }
}