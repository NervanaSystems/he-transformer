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

#include "he_tensor.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/util.hpp"
#include "seal/he_seal_backend.hpp"

ngraph::he::HETensor::HETensor(const element::Type& element_type,
                               const Shape& shape,
                               const ngraph::he::HESealBackend& he_seal_backend,
                               const bool batched, const std::string& name)
    : ngraph::runtime::Tensor(std::make_shared<ngraph::descriptor::Tensor>(
          element_type, shape, name)),
      m_he_seal_backend(he_seal_backend),
      m_batched(batched) {
  m_descriptor->set_tensor_layout(
      std::make_shared<ngraph::descriptor::layout::DenseTensorLayout>(
          *m_descriptor));

  if (batched) {
    m_batch_size = shape[0];
    m_batched_shape = batch_shape(shape, 0);
  } else {
    m_batch_size = 1;
    m_batched_shape = shape;
  }
}

ngraph::Shape ngraph::he::HETensor::batch_shape(const ngraph::Shape& shape,
                                                size_t batch_axis) {
  if (batch_axis != 0) {
    throw ngraph::ngraph_error("Batching only supported along axis 0");
  }
  ngraph::Shape batched_shape(shape);
  if (shape.size() > 0) {
    batched_shape[0] = 1;
  }
  return batched_shape;
}

void ngraph::he::HETensor::check_io_bounds(const void* source,
                                           size_t tensor_offset,
                                           size_t n) const {
  const element::Type& element_type = get_tensor_layout()->get_element_type();
  size_t type_byte_size = element_type.size();

  // Memory must be byte-aligned to type_byte_size
  // tensor_offset and n are all in bytes
  if (tensor_offset % type_byte_size != 0 || n % type_byte_size != 0) {
    throw ngraph::ngraph_error(
        "tensor_offset and n must be divisible by type_byte_size.");
  }
  // Check out-of-range
  if ((tensor_offset + n) / type_byte_size > get_element_count()) {
    throw std::out_of_range("I/O access past end of tensor");
  }
}
