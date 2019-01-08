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

#include "he_backend.hpp"
#include "he_ciphertext.hpp"
#include "he_plaintext.hpp"
#include "ngraph/log.hpp"

namespace ngraph {
namespace runtime {
namespace he {
namespace kernel {
template <typename T>
void result(const std::vector<std::shared_ptr<T>>& arg,
            std::vector<std::shared_ptr<T>>& out, size_t count) {
  if (out.size() != arg.size()) {
    NGRAPH_INFO << "Result output size " << out.size()
                << " does not match result input size " << arg.size();
    throw ngraph_error("Wrong size in result");
  }
  for (size_t i = 0; i < count; ++i) {
    out[i] = arg[i];
  }
}

void result(const std::vector<std::shared_ptr<runtime::he::HEPlaintext>>& arg,
            std::vector<std::shared_ptr<runtime::he::HECiphertext>>& out,
            size_t count, const runtime::he::HEBackend* he_backend);

void result(const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& arg,
            std::vector<std::shared_ptr<runtime::he::HEPlaintext>>& out,
            size_t count, const runtime::he::HEBackend* he_backend);
}  // namespace kernel
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
