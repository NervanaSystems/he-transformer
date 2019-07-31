//*****************************************************************************
// Copyright 2018 Intel Corporation
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

#pragma once

#include <complex>
#include <string>
#include <vector>

#include "he_tensor.hpp"
#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_cipher_tensor.hpp"

std::vector<float> read_binary_constant(const std::string filename,
                                        size_t num_elements);
std::vector<float> read_constant(const std::string filename);

// ys is logits output, or one-hot encoded ground truth
std::vector<int> batched_argmax(const std::vector<float>& ys);

void write_constant(const std::vector<float>& values,
                    const std::string filename);
void write_binary_constant(const std::vector<float>& values,
                           const std::string filename);

float get_accuracy(const std::vector<float>& pre_sigmoid,
                   const std::vector<float>& y);

template <typename T>
bool all_close(const std::vector<std::complex<T>>& a,
               const std::vector<std::complex<T>>& b,
               T atol = static_cast<T>(1e-5)) {
  for (size_t i = 0; i < a.size(); ++i) {
    if ((std::abs(a[i].real() - b[i].real()) > atol) ||
        std::abs(a[i].imag() - b[i].imag()) > atol) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool all_close(const std::vector<T>& a, const std::vector<T>& b,
               T atol = static_cast<T>(1e-3)) {
  bool close = true;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > atol) {
      NGRAPH_INFO << a[i] << " is not close to " << b[i] << " at index " << i;
      close = false;
    }
  }
  return close;
}

std::vector<std::tuple<std::vector<std::shared_ptr<ngraph::runtime::Tensor>>,
                       std::vector<std::shared_ptr<ngraph::runtime::Tensor>>>>
generate_plain_cipher_tensors(
    const std::vector<std::shared_ptr<ngraph::Node>>& output,
    const std::vector<std::shared_ptr<ngraph::Node>>& input,
    const ngraph::runtime::Backend* backend, const bool consistent_type = false,
    const bool skip_plain_plain = false);
