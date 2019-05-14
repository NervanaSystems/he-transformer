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

#include <limits>

#include "he_cipher_tensor.hpp"
#include "he_encryption_parameters.hpp"
#include "he_plain_tensor.hpp"
#include "he_tensor.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "seal/bfv/he_seal_bfv_backend.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_encryption_parameters.hpp"
#include "seal/he_seal_util.hpp"
#include "seal/seal.h"

using namespace ngraph;
using namespace std;

runtime::he::he_seal::HESealBFVBackend::HESealBFVBackend()
    : runtime::he::he_seal::HESealBFVBackend(
          runtime::he::he_seal::parse_config_or_use_default("HE_SEAL_BFV")) {}

runtime::he::he_seal::HESealBFVBackend::HESealBFVBackend(
    const shared_ptr<runtime::he::HEEncryptionParameters>& sp) {
  m_encryption_params = sp;

  auto he_seal_encryption_parms =
      static_pointer_cast<runtime::he::he_seal::HESealEncryptionParameters>(sp);
  NGRAPH_ASSERT(he_seal_encryption_parms != nullptr)
      << "HE_SEAL_BFV backend passed invalid encryption parameters";
  m_context = seal::SEALContext::Create(
      *(he_seal_encryption_parms->seal_encryption_parameters()));
  print_seal_context(*m_context);

  auto context_data = m_context->context_data();

  // Keygen, encryptor and decryptor
  m_keygen = make_shared<seal::KeyGenerator>(m_context);
  m_relin_keys = make_shared<seal::RelinKeys>(
      m_keygen->relin_keys(sp->evaluation_decomposition_bit_count()));
  m_public_key = make_shared<seal::PublicKey>(m_keygen->public_key());
  m_secret_key = make_shared<seal::SecretKey>(m_keygen->secret_key());
  m_encryptor = make_shared<seal::Encryptor>(m_context, *m_public_key);
  m_decryptor = make_shared<seal::Decryptor>(m_context, *m_secret_key);

  // Evaluator
  m_evaluator = make_shared<seal::Evaluator>(m_context);

  // Encoders
  if (context_data->qualifiers().using_batching) {
    m_batch_encoder = make_shared<seal::BatchEncoder>(m_context);
  } else {
    NGRAPH_WARN << "BFV encryption parameters not valid for batching";
  }
  m_integer_encoder = make_shared<seal::IntegerEncoder>(m_context);
}

extern "C" runtime::Backend* new_bfv_backend(const char* configuration_string) {
  return new runtime::he::he_seal::HESealBFVBackend();
}

shared_ptr<seal::SEALContext>
runtime::he::he_seal::HESealBFVBackend::make_seal_context(
    const shared_ptr<runtime::he::HEEncryptionParameters> sp) {
  throw ngraph_error("make_seal_context unimplementend");
}

namespace {
static class HESealBFVStaticInit {
 public:
  HESealBFVStaticInit() {
    runtime::BackendManager::register_backend("HE_SEAL_BFV", new_bfv_backend);
  }
  ~HESealBFVStaticInit() {}
} s_he_seal_bfv_static_init;
}  // namespace

shared_ptr<runtime::Tensor>
runtime::he::he_seal::HESealBFVBackend::create_batched_cipher_tensor(
    const element::Type& type, const Shape& shape) {
  throw ngraph_error("HESealBFVBackend::create_batched_tensor unimplemented");
}

shared_ptr<runtime::Tensor>
runtime::he::he_seal::HESealBFVBackend::create_batched_plain_tensor(
    const element::Type& type, const Shape& shape) {
  throw ngraph_error("HESealBFVBackend::create_batched_tensor unimplemented");
}

void runtime::he::he_seal::HESealBFVBackend::encode(
    shared_ptr<runtime::he::HEPlaintext>& output, const void* input,
    const element::Type& type, size_t count) const {
  if (count != 1) {
    throw ngraph_error("Batching not enabled for SEAL in encode");
  }

  NGRAPH_ASSERT(type == element::f32)
      << "BFV encode supports only float encoding, received type " << type;

  float float_val = *(float*)input;
  int32_t int_val;
  if (ceilf(float_val) == float_val) {
    int_val = static_cast<int32_t>(float_val);
  } else {
    throw ngraph_error("BFV float only supported for underlying int32_t type");
  }

  output = make_shared<runtime::he::he_seal::SealPlaintextWrapper>(
      m_integer_encoder->encode(int_val), float_val);
}

void runtime::he::he_seal::HESealBFVBackend::encode(
    runtime::he::he_seal::SealPlaintextWrapper* plaintext) const {
  std::lock_guard<std::mutex> encode_lock(m_encode_mutex);
  if (plaintext->is_encoded()) {
    return;
  }

  vector<double> double_vals(plaintext->get_values().begin(),
                             plaintext->get_values().end());

  NGRAPH_ASSERT(plaintext->num_values() == 1)
      << "BFV backend doesn't support batched encoding";

  float float_val = plaintext->get_values()[0];

  int32_t int_val;
  if (ceilf(float_val) == float_val) {
    int_val = static_cast<int32_t>(float_val);
  } else {
    throw ngraph_error("BFV float only supported for underlying int32_t type");
  }

  plaintext->get_plaintext() = m_integer_encoder->encode(int_val);
  plaintext->set_encoded(true);
}

void runtime::he::he_seal::HESealBFVBackend::decode(
    void* output, runtime::he::HEPlaintext* input, const element::Type& type,
    size_t count) const {
  if (count != 1) {
    throw ngraph_error("Batching not enabled for SEAL BFV decode");
  }
  decode(input);
  float fl_val = input->get_values()[0];
  memcpy(output, &fl_val, type.size());
}

void runtime::he::he_seal::HESealBFVBackend::decode(
    runtime::he::HEPlaintext* input) const {
  auto seal_input = dynamic_cast<const SealPlaintextWrapper*>(input);

  NGRAPH_ASSERT(seal_input != nullptr)
      << "HESealBFVBackend::decode input is not seal plaintext";

  int32_t val = m_integer_encoder->decode_int32(seal_input->get_plaintext());
  float fl_val{val};
  input->set_values({fl_val});
}