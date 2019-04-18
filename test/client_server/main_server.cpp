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

#include <iostream>
#include <memory>
#include <vector>
#include "he_backend.hpp"
#include "he_executable.hpp"
#include "ngraph/ngraph.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

int main() {
  Shape shape{2, 2};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Constant>(element::f32, shape,
                                     std::vector<float>{1.1, 1.2, 1.3, 1.4});
  auto t = (a + b) * a;
  auto f = make_shared<Function>(t, ParameterVector{a});

  NGRAPH_INFO << "Creating backend";
  auto backend = runtime::Backend::create("HE_SEAL_CKKS");

  shared_ptr<runtime::Tensor> t_a = backend->create_tensor<float>(shape);
  shared_ptr<runtime::Tensor> t_b = backend->create_tensor<float>(shape);
  shared_ptr<runtime::Tensor> t_result = backend->create_tensor<float>(shape);

  copy_data(t_a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
  copy_data(t_b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

  NGRAPH_INFO << "Compiling function";

  auto handle = backend->compile(f);

  sleep(1);
  NGRAPH_INFO << "Calling function";

  handle->call_with_validate({t_result}, {t_a});

  NGRAPH_INFO << "Sleeping";
  sleep(1);
  NGRAPH_INFO << "tearing down backend";

  return 0;
}