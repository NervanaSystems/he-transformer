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

#include "he_plaintext.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/add_seal.hpp"
#include "seal/kernel/multiply_seal.hpp"
#include "seal/kernel/subtract_seal.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph {
namespace he {
void batch_norm_inference_seal(
    double eps, std::vector<HEPlaintext>& gamma, std::vector<HEPlaintext>& beta,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& input,
    std::vector<HEPlaintext>& mean, std::vector<HEPlaintext>& variance,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& normed_input,
    const Shape& input_shape, const size_t batch_size,
    HESealBackend& he_seal_backend) {
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

    std::vector<double> channel_gamma_vals = channel_gamma.values();
    std::vector<double> channel_beta_vals = channel_beta.values();
    std::vector<double> channel_mean_vals = channel_mean.values();
    std::vector<double> channel_var_vals = channel_var.values();

    NGRAPH_CHECK(channel_gamma_vals.size() == 1, "wrong number of gamma values");
    NGRAPH_CHECK(channel_beta_vals.size() == 1, "wrong number of beta values");
    NGRAPH_CHECK(channel_mean_vals.size() == 1, "wrong number of mean values");
    NGRAPH_CHECK(channel_var_vals.size() == 1, "wrong number of var values");

    double scale = channel_gamma_vals[0] / std::sqrt(channel_var_vals[0] + eps);
    double bias =
        channel_beta_vals[0] - (channel_gamma_vals[0] * channel_mean_vals[0]) /
                                   std::sqrt(channel_var_vals[0] + eps);

    std::vector<double> scale_vec(batch_size, scale);
    std::vector<double> bias_vec(batch_size, bias);

    auto plain_scale = HEPlaintext(scale_vec);
    auto plain_bias = HEPlaintext(bias_vec);

    auto output = he_seal_backend.create_empty_ciphertext();

    scalar_multiply_seal(*input[input_index], plain_scale, output, element::f32,
                         he_seal_backend);

    scalar_add_seal(*output, plain_bias, output, element::f32, he_seal_backend);
    normed_input[input_index] = output;
  }
};
}  // namespace he
}  // namespace ngraph
