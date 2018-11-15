//*****************************************************************************
// Copyright 2018 Intel Corporation
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

#include "kernel/add.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/add_seal.hpp"

using namespace std;
using namespace ngraph::runtime::he;

void kernel::scalar_add(const shared_ptr<HECiphertext>& arg0,
                        const shared_ptr<HECiphertext>& arg1,
                        shared_ptr<HECiphertext>& out,
                        const element::Type& element_type,
                        const HEBackend* he_backend) {
  if (auto he_seal_backend =
          dynamic_cast<const he_seal::HESealBackend*>(he_backend)) {
    shared_ptr<he_seal::SealCiphertextWrapper> arg0_seal =
        dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(arg0);
    shared_ptr<he_seal::SealCiphertextWrapper> arg1_seal =
        dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(arg1);
    shared_ptr<he_seal::SealCiphertextWrapper> out_seal =
        dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(out);

    if (arg0_seal && arg1_seal && out_seal) {
      he_seal::kernel::scalar_add(arg0_seal, arg1_seal, out_seal, element_type,
                                  he_seal_backend);
      out = dynamic_pointer_cast<HECiphertext>(out_seal);
    } else {
      throw ngraph_error(
          "Add backend is SEAL, but arguments or outputs are not "
          "SealCiphertextWrapper");
    }
  } else {
    throw ngraph_error("Add backend is not SEAL.");
  }
}

void kernel::scalar_add(const shared_ptr<HEPlaintext>& arg0,
                        const shared_ptr<HEPlaintext>& arg1,
                        shared_ptr<HEPlaintext>& out,
                        const element::Type& element_type,
                        const HEBackend* he_backend) {
  if (auto he_seal_backend =
          dynamic_cast<const he_seal::HESealBackend*>(he_backend)) {
    shared_ptr<he_seal::SealPlaintextWrapper> arg0_seal =
        dynamic_pointer_cast<he_seal::SealPlaintextWrapper>(arg0);
    shared_ptr<he_seal::SealPlaintextWrapper> arg1_seal =
        dynamic_pointer_cast<he_seal::SealPlaintextWrapper>(arg1);
    shared_ptr<he_seal::SealPlaintextWrapper> out_seal =
        dynamic_pointer_cast<he_seal::SealPlaintextWrapper>(out);

    if (arg0_seal && arg1_seal && out_seal) {
      he_seal::kernel::scalar_add(arg0_seal, arg1_seal, out_seal, element_type,
                                  he_seal_backend);
      out = dynamic_pointer_cast<HEPlaintext>(out_seal);
    } else {
      throw ngraph_error(
          "Add backend is SEAL, but arguments or outputs are not "
          "SealPlaintextWrapper.:");
    }
  } else {
    throw ngraph_error("Add backend is not SEAL.");
  }
}

void kernel::scalar_add(const shared_ptr<HECiphertext>& arg0,
                        const shared_ptr<HEPlaintext>& arg1,
                        shared_ptr<HECiphertext>& out,
                        const element::Type& element_type,
                        const HEBackend* he_backend) {
  if (auto he_seal_backend =
          dynamic_cast<const he_seal::HESealBackend*>(he_backend)) {
    shared_ptr<he_seal::SealCiphertextWrapper> arg0_seal =
        dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(arg0);
    shared_ptr<he_seal::SealPlaintextWrapper> arg1_seal =
        dynamic_pointer_cast<he_seal::SealPlaintextWrapper>(arg1);
    shared_ptr<he_seal::SealCiphertextWrapper> out_seal =
        dynamic_pointer_cast<he_seal::SealCiphertextWrapper>(out);

    if (arg0_seal && arg1_seal && out_seal) {
      const string type_name = element_type.c_type_string();

      bool add_zero = he_backend->optimized_add();
      if (add_zero) {
        auto seal_0_plaintext =
            static_pointer_cast<const he_seal::SealPlaintextWrapper>(
                he_seal_backend->get_valued_plaintext(0))
                ->m_plaintext;

        add_zero = (arg1_seal->m_plaintext == seal_0_plaintext);
      }

      if (add_zero && type_name == "float") {
        NGRAPH_INFO << "Optimized add by 0";
        out = arg0;
      } else {
        he_seal::kernel::scalar_add(arg0_seal, arg1_seal, out_seal,
                                    element_type, he_seal_backend);
        out = dynamic_pointer_cast<HECiphertext>(out_seal);
      }
    } else {
      throw ngraph_error(
          "Add backend is SEAL, but arguments or outputs are not "
          "SealPlaintextWrapper");
    }
  } else {
    throw ngraph_error("Add backend is not SEAL.");
  }
}

void kernel::scalar_add(const shared_ptr<HEPlaintext>& arg0,
                        const shared_ptr<HECiphertext>& arg1,
                        shared_ptr<HECiphertext>& out,
                        const element::Type& element_type,
                        const HEBackend* he_backend) {
  scalar_add(arg1, arg0, out, element_type, he_backend);
}
