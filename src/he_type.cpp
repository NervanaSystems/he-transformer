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

#include "he_type.hpp"

#include <memory>

#include "he_plaintext.hpp"
#include "ngraph/type/element_type.hpp"
#include "protos/message.pb.h"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph::he {

HEType HEType::load(const proto::HEType& proto_he_type,
                    std::shared_ptr<seal::SEALContext> context) {
  if (proto_he_type.is_plaintext()) {
    // TODO(fboemer): HEPlaintext::load function
    HEPlaintext vals{proto_he_type.plain().begin(),
                     proto_he_type.plain().end()};

    return HEType(vals, proto_he_type.complex_packing());
  }
  auto cipher = HESealBackend::create_empty_ciphertext();
  SealCiphertextWrapper::load(*cipher, proto_he_type, std::move(context));

  return HEType(cipher, proto_he_type.complex_packing(),
                proto_he_type.batch_size());
}

}  // namespace ngraph::he
