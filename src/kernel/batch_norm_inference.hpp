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

namespace ngraph {
namespace runtime {
namespace he {
namespace kernel {
template <typename S, typename T>
void batch_norm_inference(double eps,
                          const std::vector<std::shared_ptr<T>>& gamma,
                          const std::vector<std::shared_ptr<T>>& beta,
                          const std::vector<std::shared_ptr<S>>& input,
                          const std::vector<std::shared_ptr<T>>& mean,
                          const std::vector<std::shared_ptr<T>>& variance,
                          std::vector<std::shared_ptr<S>>& normed_input,
                          const Shape& input_shape,
                          const HEBackend* he_backend) {
  CoordinateTransform input_transform(input_shape);

  // Store input coordinates for parallelization
  std::vector<ngraph::Coordinate> input_coords;
  for (const Coordinate& in_coord : input_transform) {
    input_coords.emplace_back(in_coord);
  }
  size_t input_transform_size = input_coords.size();

  NGRAPH_INFO << "input_transform_size" << input_transform_size;

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

    std::shared_ptr<S> normalized = he_backend->create_empty_hetext<S>(S{});
    // std::shared_ptr<S> output = he_backend->create_empty_hetext<S>(S{});

    // auto normalized = (input[input_index] - channel_mean) /
    //                  (std::sqrt(channel_var + eps_casted));
    // normed_input[input_index] = normalized * channel_gamma + channel_beta;

    // NGRAPH_INFO << "Subtracting";
    // TODO: replace with subtract with proper scale
    runtime::he::kernel::scalar_add(input[input_index].get(),
                                    channel_mean.get(), normalized,
                                    element::f32, he_backend);
    // To simulate multiplicative depth, multiply instead of divide
    // TODO: make values correct.
    // NGRAPH_INFO << "Multiplying";
    runtime::he::kernel::scalar_multiply(normalized.get(), channel_var.get(),
                                         normalized, element::f32, he_backend);

    // NGRAPH_INFO << "Multiplying again";
    runtime::he::kernel::scalar_multiply(normalized.get(), channel_gamma.get(),
                                         normalized, element::f32, he_backend);
    // NGRAPH_INFO << "Adding";
    runtime::he::kernel::scalar_add(normalized.get(), channel_beta.get(),
                                    normalized, element::f32, he_backend);

    normed_input[input_index] = normalized;
  }
};
}  // namespace kernel
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
