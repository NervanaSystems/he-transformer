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

#include "kernel/result.hpp"

void ngraph::he::result(
    const std::vector<std::unique_ptr<ngraph::he::HEPlaintext>>& arg,
    std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>& out,
    size_t count, const ngraph::he::HESealBackend* he_seal_backend) {
  if (out.size() != arg.size()) {
    NGRAPH_INFO << "Result output size " << out.size()
                << " does not match result input size " << arg.size();
    throw ngraph_error("Wrong size in result");
  }
  for (size_t i = 0; i < count; ++i) {
    he_seal_backend->encrypt(out[i], *arg[i]);
  }
}

void ngraph::he::result(
    const std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>& arg,
    std::vector<std::unique_ptr<ngraph::he::HEPlaintext>>& out, size_t count,
    const ngraph::he::HESealBackend* he_seal_backend) {
  if (out.size() != arg.size()) {
    NGRAPH_INFO << "Result output size " << out.size()
                << " does not match result input size " << arg.size();
    throw ngraph_error("Wrong size in result");
  }
  for (size_t i = 0; i < count; ++i) {
    he_seal_backend->decrypt(*out[i], arg[i]);
    // he_seal_backend->decode(*out[i]);
  }
}
