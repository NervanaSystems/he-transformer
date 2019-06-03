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

#include <memory>
#include <vector>

#include "kernel/add.hpp"
#include "kernel/negate.hpp"
#include "kernel/subtract.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/subtract_seal.hpp"

void ngraph::he::scalar_subtract(std::shared_ptr<SealCiphertextWrapper>& arg0,
                                 std::shared_ptr<SealCiphertextWrapper>& arg1,
                                 std::shared_ptr<SealCiphertextWrapper>& out,
                                 const element::Type& element_type,
                                 const ngraph::he::HESealBackend* he_seal_backend) {
  auto he_seal_backend = ngraph::he::cast_to_seal_backend(he_seal_backend);
  auto arg0_seal = ngraph::he::cast_to_seal_hetext(arg0);
  auto arg1_seal = ngraph::he::cast_to_seal_hetext(arg1);
  auto out_seal = ngraph::he::cast_to_seal_hetext(out);

  ngraph::he::scalar_subtract(arg0_seal, arg1_seal, out_seal, element_type,
                              he_seal_backend);
  out = std::dynamic_pointer_cast<SealCiphertextWrapper>(out_seal);
}

void ngraph::he::scalar_subtract(const HEPlaintext& arg0,
                                 const HEPlaintext& arg1, HEPlaintext& out,
                                 const element::Type& element_type,
                                 const ngraph::he::HESealBackend* he_seal_backend) {
  NGRAPH_CHECK(element_type == element::f32);

  std::vector<float> arg0_vals = arg0.get_values();
  std::vector<float> arg1_vals = arg1.get_values();
  std::vector<float> out_vals(arg0.num_values());

  std::transform(arg0_vals.begin(), arg0_vals.end(), arg1_vals.begin(),
                 out_vals.begin(), std::minus<float>());

  out.set_values(out_vals);
}

void ngraph::he::scalar_subtract(std::shared_ptr<SealCiphertextWrapper>& arg0,
                                 const HEPlaintext& arg1,
                                 std::shared_ptr<SealCiphertextWrapper>& out,
                                 const element::Type& type,
                                 const ngraph::he::HESealBackend* he_seal_backend) {
  NGRAPH_CHECK(type == element::f32);
  NGRAPH_INFO << "cipher - plain";

  if (arg0->is_zero()) {
    NGRAPH_INFO << "arg0 is zero";
    HEPlaintext tmp;
    ngraph::he::scalar_negate(arg1, tmp, type);
    he_seal_backend->encrypt(out, tmp);

    return;
  }

  auto he_seal_backend = ngraph::he::cast_to_seal_backend(he_seal_backend);
  auto arg0_seal = ngraph::he::cast_to_seal_hetext(arg0);
  auto out_seal = ngraph::he::cast_to_seal_hetext(out);

  bool sub_zero = arg1.is_single_value() && (arg1.get_values()[0] == 0.0f);

  if (sub_zero) {
    // Make copy of input
    // TODO: make copy only if necessary
    NGRAPH_INFO << "Sub 0 optimization";
    out = std::static_pointer_cast<SealCiphertextWrapper>(
        std::make_shared<ngraph::he::SealCiphertextWrapper>(*arg0_seal));
  } else {
    NGRAPH_INFO << "normal sub";
    scalar_subtract(arg0_seal, arg1, out_seal, type, he_seal_backend);
  }
}

void ngraph::he::scalar_subtract(const HEPlaintext& arg0,
                                 std::shared_ptr<SealCiphertextWrapper>& arg1,
                                 std::shared_ptr<SealCiphertextWrapper>& out,
                                 const element::Type& type,
                                 const ngraph::he::HESealBackend* he_seal_backend) {
  if (arg1->is_zero()) {
    he_seal_backend->encrypt(out, arg0);
  } else {
    ngraph::he::scalar_negate(arg1, out, type, he_seal_backend);
    ngraph::he::scalar_add(arg0, out, out, type, he_seal_backend);
  }
}
