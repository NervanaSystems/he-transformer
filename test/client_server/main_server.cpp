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
#include "ngraph/ngraph.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"

using namespace ngraph;
using namespace std;

int main() {
  Shape shape{2, 3};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Constant>(
      element::f32, shape, std::vector<float>{1.1, 1.2, 1.3, 1.4, 1.5, 1.6});
  auto t = make_shared<op::Add>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{a});

  NGRAPH_INFO << "Creating backend";
  auto backend = runtime::Backend::create("HE_SEAL_CKKS");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  auto handle = backend->compile(f);

  NGRAPH_INFO << "Starting server";

  he_backend->start_server();

  return 0;
}