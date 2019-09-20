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

#include <assert.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "test_util.hpp"

std::vector<std::tuple<std::vector<std::shared_ptr<ngraph::runtime::Tensor>>,
                       std::vector<std::shared_ptr<ngraph::runtime::Tensor>>>>
generate_plain_cipher_tensors(
    const std::vector<std::shared_ptr<ngraph::Node>>& output,
    const std::vector<std::shared_ptr<ngraph::Node>>& input,
    const ngraph::runtime::Backend* backend, const bool consistent_type,
    const bool skip_plain_plain) {
  auto he_backend = static_cast<const ngraph::he::HESealBackend*>(backend);

  using TupleOfInputOutputs = std::vector<
      std::tuple<std::vector<std::shared_ptr<ngraph::runtime::Tensor>>,
                 std::vector<std::shared_ptr<ngraph::runtime::Tensor>>>>;
  TupleOfInputOutputs ret;

  auto cipher_cipher = [&output, &input, &he_backend]() {
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> result;
    for (auto elem : output) {
      auto output_tensor = he_backend->create_cipher_tensor(
          elem->get_element_type(), elem->get_shape());
      result.push_back(output_tensor);
    }
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> argument;
    for (auto elem : input) {
      auto input_tensor = he_backend->create_cipher_tensor(
          elem->get_element_type(), elem->get_shape());
      argument.push_back(input_tensor);
    }
    return std::make_tuple(result, argument);
  };

  auto plain_plain = [&output, &input, &he_backend]() {
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> result;
    for (auto elem : output) {
      auto output_tensor = he_backend->create_plain_tensor(
          elem->get_element_type(), elem->get_shape());
      result.push_back(output_tensor);
    }
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> argument;
    for (auto elem : input) {
      auto input_tensor = he_backend->create_plain_tensor(
          elem->get_element_type(), elem->get_shape());
      argument.push_back(input_tensor);
    }
    return std::make_tuple(result, argument);
  };
  auto alternate_cipher = [&output, &input, &he_backend](size_t mod) {
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> result;
    for (auto elem : output) {
      auto output_tensor = he_backend->create_cipher_tensor(
          elem->get_element_type(), elem->get_shape());
      result.push_back(output_tensor);
    }
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> argument;
    for (size_t i = 0; i < input.size(); ++i) {
      auto elem = input[i];
      if (i % 2 == mod) {
        auto input_tensor = he_backend->create_plain_tensor(
            elem->get_element_type(), elem->get_shape());
        argument.push_back(input_tensor);
      } else {
        auto input_tensor = he_backend->create_cipher_tensor(
            elem->get_element_type(), elem->get_shape());
        argument.push_back(input_tensor);
      }
    }
    return std::make_tuple(result, argument);
  };
  auto plain_cipher_cipher = [&alternate_cipher]() {
    return alternate_cipher(0);
  };
  auto cipher_plain_cipher = [&alternate_cipher]() {
    return alternate_cipher(1);
  };

  if (he_backend != nullptr) {
    if (!skip_plain_plain) {
      ret.push_back(plain_plain());
    }
    if (!consistent_type) {
      ret.push_back(plain_cipher_cipher());
    }
    if (input.size() >= 2 && !consistent_type) {
      ret.push_back(cipher_plain_cipher());
    }
    ret.push_back(cipher_cipher());
  }
  return ret;
}
