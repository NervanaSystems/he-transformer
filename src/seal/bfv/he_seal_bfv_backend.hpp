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

#include "he_cipher_tensor.hpp"
#include "he_plain_tensor.hpp"
#include "he_tensor.hpp"
#include "ngraph/runtime/backend.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_encryption_parameters.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace runtime {
namespace he {
namespace he_seal {
class HESealBFVBackend : public HESealBackend {
 public:
  HESealBFVBackend();
  HESealBFVBackend(
      const std::shared_ptr<runtime::he::HEEncryptionParameters>& sp);
  HESealBFVBackend(HESealBFVBackend& he_backend) = default;
  ~HESealBFVBackend(){};

  std::shared_ptr<runtime::Tensor> create_batched_cipher_tensor(
      const element::Type& element_type, const Shape& shape) override;

  std::shared_ptr<runtime::Tensor> create_batched_plain_tensor(
      const element::Type& element_type, const Shape& shape) override;

  std::shared_ptr<seal::SEALContext> make_seal_context(
      const std::shared_ptr<runtime::he::HEEncryptionParameters> sp) override;

  void encode(
      runtime::he::he_seal::SealPlaintextWrapper* plaintext) const override;

  void encode(
      std::vector<std::shared_ptr<runtime::he::he_seal::SealPlaintextWrapper>>&
          plaintexts) const override {
    throw ngraph_error("Unimplemented");
  }

  void encode(std::shared_ptr<runtime::he::HEPlaintext>& output,
              const void* input, const element::Type& element_type,
              size_t count = 1) const override;
  void decode(void* output, const runtime::he::HEPlaintext* input,
              const element::Type& element_type,
              size_t count = 1) const override;

  const inline std::shared_ptr<seal::BatchEncoder> get_batch_encoder() const {
    return m_batch_encoder;
  }

  const inline std::shared_ptr<seal::IntegerEncoder> get_integer_encoder()
      const {
    return m_integer_encoder;
  }

 private:
  std::shared_ptr<seal::BatchEncoder> m_batch_encoder;
  std::shared_ptr<seal::IntegerEncoder> m_integer_encoder;
  mutable std::mutex m_encode_mutex;
};
}  // namespace he_seal
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
