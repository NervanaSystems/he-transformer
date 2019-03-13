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
namespace runtime {
namespace he {
namespace kernel {
void batch_norm_inference(
    double eps, const std::vector<std::shared_ptr<HEPlaintext>>& gamma,
    const std::vector<std::shared_ptr<HEPlaintext>>& beta,
    const std::vector<std::shared_ptr<HECiphertext>>& input,
    const std::vector<std::shared_ptr<HEPlaintext>>& mean,
    const std::vector<std::shared_ptr<HEPlaintext>>& variance,
    std::vector<std::shared_ptr<HECiphertext>>& normed_input,
    const Shape& input_shape, const HEBackend* he_backend) {
  CoordinateTransform input_transform(input_shape);

  auto he_seal_backend =
      dynamic_cast<const runtime::he::he_seal::HESealBackend*>(he_backend);

  if (he_seal_backend == nullptr) {
    throw ngraph_error(
        "BatchNormInference unimplemented for non-seal backends");
  }

  // Store input coordinates for parallelization
  std::vector<ngraph::Coordinate> input_coords;
  for (const Coordinate& in_coord : input_transform) {
    input_coords.emplace_back(in_coord);
  }
  size_t input_transform_size = input_coords.size();

  NGRAPH_DEBUG << "input_transform_size " << input_transform_size;

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

    std::vector<float> channel_gamma_fl(1, 0);
    std::vector<float> channel_beta_fl(1, 0);
    std::vector<float> channel_mean_fl(1, 0);
    std::vector<float> channel_var_fl(1, 0);

    he_seal_backend->decode((void*)(channel_gamma_fl.data()),
                            channel_gamma.get(), element::f32, 1);
    he_seal_backend->decode((void*)(channel_beta_fl.data()), channel_beta.get(),
                            element::f32, 1);
    he_seal_backend->decode((void*)(channel_mean_fl.data()), channel_mean.get(),
                            element::f32, 1);
    he_seal_backend->decode((void*)(channel_var_fl.data()), channel_var.get(),
                            element::f32, 1);

    NGRAPH_DEBUG << "channel_gamma_fl " << channel_gamma_fl[0];
    NGRAPH_DEBUG << "channel_beta_fl " << channel_beta_fl[0];
    NGRAPH_DEBUG << "channel_mean_fl " << channel_mean_fl[0];
    NGRAPH_DEBUG << "channel_var_fl " << channel_var_fl[0];

    float scale = channel_gamma_fl[0] / std::sqrt(channel_var_fl[0] + eps);
    float bias = channel_beta_fl[0] -
                 channel_mean_fl[0] / std::sqrt(channel_var_fl[0] + eps);

    NGRAPH_DEBUG << "scale " << scale;
    NGRAPH_DEBUG << "bias " << bias;

    auto plain_scale = he_backend->create_empty_plaintext();
    he_seal_backend->encode(plain_scale, &scale, element::f32);

    auto plain_bias = he_backend->create_empty_plaintext();
    he_seal_backend->encode(plain_bias, &bias, element::f32);

    std::shared_ptr<HECiphertext> output =
        he_backend->create_empty_ciphertext();

    runtime::he::kernel::scalar_multiply(input[input_index].get(),
                                         plain_scale.get(), output,
                                         element::f32, he_backend);

    runtime::he::kernel::scalar_add(output.get(), plain_bias.get(), output,
                                    element::f32, he_backend);

    // auto normalized = (input[input_index] - channel_mean) /
    //                  (std::sqrt(channel_var + eps_casted));
    // normed_input[input_index] = normalized * channel_gamma +
    // channel_beta;

    // std::shared_ptr<S> output = he_backend->create_empty_hetext<S>(S{});

    // runtime::he::kernel::scalar_multiply(input[input_index].get(),
    //                                     channel_var.get(), normalized,
    //                                     element::f32, he_backend);

    // runtime::he::kernel::scalar_add(normalized.get(), channel_mean.get(),
    //                               normalized, element::f32, he_backend);
    // To simulate multiplicative depth, multiply instead of divide
    // TODO: make values correct.
    // NGRAPH_DEBUG << "Multiplying";

    // NGRAPH_DEBUG << "Multiplying again";
    // runtime::he::kernel::scalar_multiply(normalized.get(),
    // channel_gamma.get(),
    //   normalized, element::f32, he_backend);
    // NGRAPH_DEBUG << "Adding";
    // runtime::he::kernel::scalar_add(normalized.get(), channel_beta.get(),
    //                                normalized, element::f32, he_backend);

    normed_input[input_index] = output;
  }
};
}  // namespace kernel
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
