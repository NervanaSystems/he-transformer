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

#include "he_op_annotations.hpp"
#include "ngraph/ngraph.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/add_seal.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::he;

static string s_manifest = "${MANIFEST}";

TEST(add, mod_wrap) {
  auto backend = runtime::Backend::create("HE_SEAL");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  // Use
  // NGRAPH_HE_SEAL_CONFIG=$HE_TRANSFORMER/configs/he_seal_ckks_config_N11_L1_small.json

  // No modulus wrap
  for (size_t i = 1; i < 1024; i += 1) {
    HEPlaintext plain(std::vector<double>(i, -24.5698));
    HEPlaintext mask(std::vector<double>(i, -31.4277));

    // No modulus wrap
    // HEPlaintext plain({-5.37476, -24.5698});
    // HEPlaintext mask({-31.4277, -31.4277});

    // No modulus wrap
    // HEPlaintext plain({-5.37476, -24.5698});
    // HEPlaintext mask({-31.4277, -31.4277});

    // No modulus wrapping
    // HEPlaintext plain({-5.37476, -24.5698});
    // HEPlaintext mask({5.37476, -31.4277});

    // Modulus wrapping as expected
    // HEPlaintext plain({-24.5698});
    // HEPlaintext mask({-31.4277});

    auto cipher = HESealBackend::create_empty_ciphertext();

    he_backend->encrypt(cipher, plain, element::f32);

    scalar_add_seal(*cipher, mask, cipher, false, *he_backend);

    he_backend->decrypt(plain, *cipher, false);

    plain = HEPlaintext(std::vector<double>{plain.begin(), plain.begin() + 2});

    NGRAPH_INFO << "i " << i << " plain " << plain[0];
  }
}