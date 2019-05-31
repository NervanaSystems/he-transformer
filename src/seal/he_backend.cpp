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

#include <chrono>
#include <limits>
#include <memory>
#include <thread>

#include "he_backend.hpp"
#include "he_cipher_tensor.hpp"
#include "he_executable.hpp"
#include "he_plain_tensor.hpp"
#include "he_tensor.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/function.hpp"

using ngraph::descriptor::layout::DenseTensorLayout;

std::shared_ptr<ngraph::he::HECiphertext>
ngraph::he::HEBackend::create_valued_ciphertext(
    float value, const element::Type& element_type, size_t batch_size) const {
  NGRAPH_CHECK(element_type == element::f32, "element type ", element_type,
               "unsupported");
  if (batch_size != 1) {
    throw ngraph_error(
        "HEBackend::create_valued_ciphertext only supports batch size 1");
  }
  std::unique_ptr<ngraph::he::HEPlaintext> plaintext =
      create_valued_plaintext({value}, m_complex_packing);
  std::shared_ptr<ngraph::he::HECiphertext> ciphertext =
      create_empty_ciphertext();

  encrypt(ciphertext, *plaintext);
  return ciphertext;
}

std::shared_ptr<ngraph::runtime::Tensor> ngraph::he::HEBackend::create_tensor(
    const element::Type& element_type, const Shape& shape,
    void* memory_pointer) {
  throw ngraph_error("HE create_tensor unimplemented");
}

std::shared_ptr<ngraph::runtime::Tensor> ngraph::he::HEBackend::create_tensor(
    const element::Type& element_type, const Shape& shape) {
  if (batch_data()) {
    return create_batched_plain_tensor(element_type, shape);
  } else {
    return create_plain_tensor(element_type, shape);
  }
}

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HEBackend::create_plain_tensor(const element::Type& element_type,
                                           const Shape& shape,
                                           const bool batched) const {
  auto rc = std::make_shared<ngraph::he::HEPlainTensor>(
      element_type, shape, this, create_empty_plaintext(), batched);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(rc);
}

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HEBackend::create_cipher_tensor(const element::Type& element_type,
                                            const Shape& shape,
                                            const bool batched) const {
  auto rc = std::make_shared<ngraph::he::HECipherTensor>(
      element_type, shape, this, create_empty_ciphertext(), batched);
  return std::static_pointer_cast<ngraph::runtime::Tensor>(rc);
}

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HEBackend::create_valued_cipher_tensor(
    float value, const element::Type& element_type, const Shape& shape) const {
  auto tensor = std::static_pointer_cast<HECipherTensor>(
      create_cipher_tensor(element_type, shape));
  std::vector<std::shared_ptr<ngraph::he::HECiphertext>>& cipher_texts =
      tensor->get_elements();
#pragma omp parallel for
  for (size_t i = 0; i < cipher_texts.size(); ++i) {
    cipher_texts[i] = create_valued_ciphertext(value, element_type);
  }
  return tensor;
}

std::shared_ptr<ngraph::runtime::Tensor>
ngraph::he::HEBackend::create_valued_plain_tensor(
    float value, const element::Type& element_type, const Shape& shape) const {
  auto tensor = std::static_pointer_cast<HEPlainTensor>(
      create_plain_tensor(element_type, shape));
  //#pragma omp parallel for
  // TODO: check it's fast
  for (size_t i = 0; i < tensor->num_plaintexts(); ++i) {
    tensor->get_element(i) =
        std::move(create_valued_plaintext({value}, m_complex_packing));
  }
  return tensor;
}

std::shared_ptr<ngraph::runtime::Executable> ngraph::he::HEBackend::compile(
    std::shared_ptr<Function> function, bool enable_performance_collection) {
  return std::make_shared<HEExecutable>(function, enable_performance_collection,
                                        this, m_encrypt_data, m_encrypt_model,
                                        m_batch_data);
}
