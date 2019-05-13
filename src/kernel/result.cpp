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

using namespace std;
using namespace ngraph;

// TODO: remove
void runtime::he::kernel::result(
    const vector<shared_ptr<runtime::he::HEPlaintext>>& arg,
    vector<shared_ptr<runtime::he::HECiphertext>>& out, size_t count,
    const runtime::he::HEBackend* he_backend) {
  if (out.size() != arg.size()) {
    NGRAPH_INFO << "Result output size " << out.size()
                << " does not match result input size " << arg.size();
    throw ngraph_error("Wrong size in result");
  }
  for (size_t i = 0; i < count; ++i) {
    he_backend->encrypt(out[i], arg[i].get());
  }
}

void runtime::he::kernel::result(
    const vector<shared_ptr<runtime::he::HECiphertext>>& arg,
    vector<shared_ptr<runtime::he::HEPlaintext>>& out, size_t count,
    const runtime::he::HEBackend* he_backend) {
  if (out.size() != arg.size()) {
    NGRAPH_INFO << "Result output size " << out.size()
                << " does not match result input size " << arg.size();
    throw ngraph_error("Wrong size in result");
  }
  for (size_t i = 0; i < count; ++i) {
    he_backend->decrypt(out[i], arg[i].get());
    NGRAPH_INFO << "Decoding result " << i;
    he_backend->decode(out[i].get());
    NGRAPH_INFO << "Done decoding result " << i;
  }
}