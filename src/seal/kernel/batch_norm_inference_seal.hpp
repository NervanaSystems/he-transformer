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

#include "he_type.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/add_seal.hpp"
#include "seal/kernel/multiply_seal.hpp"
#include "seal/kernel/subtract_seal.hpp"

namespace ngraph {
namespace he {
void batch_norm_inference_seal(
    double eps, std::vector<HEType>& gamma, std::vector<HEType>& beta,
    std::vector<HEType>& input, std::vector<HEType>& mean,
    std::vector<HEType>& variance, std::vector<HEType>& normed_input,
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

    NGRAPH_CHECK(channel_gamma.is_plaintext(),
                 "BatchNorm inference only supportd for plaintext");
    NGRAPH_CHECK(channel_beta.is_plaintext(),
                 "BatchNorm inference only supportd for plaintext");
    NGRAPH_CHECK(channel_mean.is_plaintext(),
                 "BatchNorm inference only supportd for plaintext");
    NGRAPH_CHECK(channel_var.is_plaintext(),
                 "BatchNorm inference only supportd for plaintext");

    HEPlaintext channel_gamma_vals = channel_gamma.get_plaintext();
    HEPlaintext channel_beta_vals = channel_beta.get_plaintext();
    HEPlaintext channel_mean_vals = channel_mean.get_plaintext();
    HEPlaintext channel_var_vals = channel_var.get_plaintext();

    NGRAPH_CHECK(channel_gamma_vals.size() == 1,
                 "wrong number of gamma values");
    NGRAPH_CHECK(channel_beta_vals.size() == 1, "wrong number of beta values");
    NGRAPH_CHECK(channel_mean_vals.size() == 1, "wrong number of mean values");
    NGRAPH_CHECK(channel_var_vals.size() == 1, "wrong number of var values");

    double scale = channel_gamma_vals[0] / std::sqrt(channel_var_vals[0] + eps);
    double bias =
        channel_beta_vals[0] - (channel_gamma_vals[0] * channel_mean_vals[0]) /
                                   std::sqrt(channel_var_vals[0] + eps);

    HEPlaintext scale_vec(std::vector<double>(batch_size, scale));
    HEPlaintext bias_vec(std::vector<double>(batch_size, bias));

    HEType he_scale(scale_vec, false, false, batch_size);
    HEType he_bias(bias_vec, false, false, batch_size);

    scalar_multiply_seal(input[input_index], he_scale,
                         normed_input[input_index], he_seal_backend);

    scalar_add_seal(normed_input[input_index], he_bias,
                    normed_input[input_index], he_seal_backend);
  }
};
}  // namespace he
}  // namespace ngraph
