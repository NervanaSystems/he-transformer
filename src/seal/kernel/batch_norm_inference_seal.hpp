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

#pragma once

#include <memory>
#include <vector>

#include "he_ciphertext.hpp"
#include "he_plaintext.hpp"
#include "kernel/add.hpp"
#include "kernel/multiply.hpp"
#include "kernel/subtract.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace he {
void batch_norm_inference(
    double eps, std::vector<std::shared_ptr<HEPlaintext>>& gamma,
    std::vector<std::shared_ptr<HEPlaintext>>& beta,
    std::vector<std::shared_ptr<HECiphertext>>& input,
    std::vector<std::shared_ptr<HEPlaintext>>& mean,
    std::vector<std::shared_ptr<HEPlaintext>>& variance,
    std::vector<std::shared_ptr<HECiphertext>>& normed_input,
    const Shape& input_shape, const size_t batch_size,
    const HESealBackend* he_seal_backend) {
  CoordinateTransform input_transform(input_shape);

  // Store input coordinates for parallelization
  std::vector<ngraph::Coordinate> input_coords;
  for (const Coordinate& in_coord : input_transform) {
    input_coords.emplace_back(in_coord);
  }
  size_t input_transform_size = input_coords.size();

#pragma omp parallel for
  for (size_t i = 0; i < input_transform_size; ++i) {
    Coordinate input_coord = input_coords[i];
    // for (Coordinate input_coord : input_transform) {
    auto channel_num = input_coord[1];
    auto channel_gamma = gamma[channel_num];
    auto channel_beta = beta[channel_num];
    auto channel_mean = mean[channel_num];
    auto channel_var = variance[channel_num];

    auto input_index = input_transform.index(input_coord);

    std::vector<float> channel_gamma_vals = channel_gamma->get_values();
    std::vector<float> channel_beta_vals = channel_beta->get_values();
    std::vector<float> channel_mean_vals = channel_mean->get_values();
    std::vector<float> channel_var_vals = channel_var->get_values();

    NGRAPH_CHECK(channel_gamma_vals.size() == 1);
    NGRAPH_CHECK(channel_beta_vals.size() == 1);
    NGRAPH_CHECK(channel_mean_vals.size() == 1);
    NGRAPH_CHECK(channel_var_vals.size() == 1);

    float scale = channel_gamma_vals[0] / std::sqrt(channel_var_vals[0] + eps);
    float bias =
        channel_beta_vals[0] - (channel_gamma_vals[0] * channel_mean_vals[0]) /
                                   std::sqrt(channel_var_vals[0] + eps);

    std::vector<float> scale_vec(batch_size, scale);
    std::vector<float> bias_vec(batch_size, bias);

    // TODO: enable complex packing?
    auto plain_scale =
        create_valued_plaintext(std::vector<float>{scale_vec}, false);
    auto plain_bias =
        create_valued_plaintext(std::vector<float>{bias_vec}, false);

    auto output = he_seal_backend->create_empty_ciphertext();

    ngraph::he::scalar_multiply(input[input_index], *plain_scale, output,
                                element::f32, he_seal_backend);

    ngraph::he::scalar_add(output, *plain_bias, output, element::f32,
                           he_seal_backend);
    normed_input[input_index] = output;
  }
};
}  // namespace he
}  // namespace ngraph
