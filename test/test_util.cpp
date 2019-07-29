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

std::vector<float> read_constant(const std::string filename) {
  std::string data = ngraph::file_util::read_file_to_string(filename);
  std::stringstream iss(data);

  std::vector<std::string> constants;
  std::copy(std::istream_iterator<std::string>(iss),
            std::istream_iterator<std::string>(), back_inserter(constants));

  std::vector<float> res;
  for (const std::string& constant : constants) {
    res.push_back(atof(constant.c_str()));
  }
  return res;
}

std::vector<int> batched_argmax(const std::vector<float>& ys) {
  if (ys.size() % 10 != 0) {
    std::cout << "ys.size() must be a multiple of 10" << std::endl;
    exit(1);
  }
  std::vector<int> labels;
  const float* data = ys.data();
  size_t idx = 0;
  while (idx < ys.size()) {
    int label = std::distance(data + idx,
                              std::max_element(data + idx, data + idx + 10));
    labels.push_back(label);
    idx += 10;
  }
  return labels;
}

float get_accuracy(const std::vector<float>& pre_sigmoid,
                   const std::vector<float>& y) {
  assert(pre_sigmoid.size() % 10 == 0);
  size_t num_data = pre_sigmoid.size() / 10;

  size_t correct = 0;
  for (size_t i = 0; i < num_data; ++i) {
    std::vector<float> sub_vec(pre_sigmoid.begin() + i * 10,
                               pre_sigmoid.begin() + (i + 1) * 10);
    auto minmax = minmax_element(sub_vec.begin(), sub_vec.end());
    size_t prediction = minmax.second - sub_vec.begin();

    if (round(y[10 * i + prediction]) == 1) {
      correct++;
    }
  }
  return correct / float(num_data);
}

std::vector<float> read_binary_constant(const std::string filename,
                                        size_t num_elements) {
  std::ifstream infile;
  std::vector<float> values(num_elements);
  infile.open(filename, std::ios::in | std::ios::binary);
  infile.read(reinterpret_cast<char*>(&values[0]),
              num_elements * sizeof(float));
  infile.close();
  return values;
}

void write_binary_constant(const std::vector<float>& values,
                           const std::string filename) {
  std::ofstream outfile(filename, std::ios::out | std::ios::binary);
  outfile.write(reinterpret_cast<const char*>(&values[0]),
                values.size() * sizeof(float));
  outfile.close();
}

std::vector<std::tuple<std::vector<std::shared_ptr<runtime::Tensor>>,
                       std::vector<std::shared_ptr<runtime::Tensor>>>>
generate_plain_cipher_tensors(const std::vector<std::shared_ptr<Node>>& output,
                              const std::vector<std::shared_ptr<Node>>& input,
                              const ngraph::runtime::Backend* backend,
                              const bool consistent_type,
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
