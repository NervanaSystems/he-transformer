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

#include "ngraph/ngraph.hpp"
#include "seal/seal.h"
#include "seal/seal_util.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
using namespace he;
using namespace seal;

static string s_manifest = "${MANIFEST}";

TEST(perf_micro, encode) {
  auto perf_test = [](size_t poly_modulus_degree,
                      const std::vector<int>& coeff_modulus_bits) {
    int add_test_cnt = 1000;
    int mult_test_cnt = 300;
    int encode_test_cnt = 1000;
    int max_test_count = max(max(add_test_cnt, mult_test_cnt), encode_test_cnt);

    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::nanoseconds time_seal_encode_sum(0);
    chrono::nanoseconds time_he_encode_sum(0);

    chrono::nanoseconds time_seal_multiply_plain_sum(0);
    chrono::nanoseconds time_he_multiply_plain_sum(0);

    chrono::nanoseconds time_seal_add_plain_sum(0);
    chrono::nanoseconds time_he_add_plain_sum(0);

    // Seal setup
    EncryptionParameters parms(scheme_type::CKKS);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(
        CoeffModulus::Create(poly_modulus_degree, coeff_modulus_bits));

    auto context = SEALContext::Create(parms);
    CKKSEncoder encoder(context);
    KeyGenerator keygen(context);
    auto secret_key = keygen.secret_key();
    auto public_key = keygen.public_key();
    Encryptor encryptor(context, public_key);
    Decryptor decryptor(context, secret_key);
    Evaluator evaluator(context);

    // he-transformer setup
    auto he_parms = HESealEncryptionParameters("HE_SEAL", poly_modulus_degree,
                                               128, coeff_modulus_bits);
    auto he_seal_backend = HESealBackend(he_parms);

    for (int test_run = 0; test_run < max_test_count; ++test_run) {
      auto seal_capacity =
          poly_modulus_degree * (coeff_modulus_bits.size() - 1);
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool();

      Plaintext plain(seal_capacity);
      double input{1.23};
      double scale = pow(2.0, 22);
      std::vector<std::uint64_t> he_plain;
      auto parms_id = context->first_parms_id();

      // [Encoding]
      if (test_run < encode_test_cnt) {
        // SEAL encoder
        time_start = chrono::high_resolution_clock::now();
        encoder.encode(input, scale, plain, pool);
        encoder.encode(input, scale, plain, pool);
        encoder.encode(input, scale, plain, pool);
        time_end = chrono::high_resolution_clock::now();
        time_seal_encode_sum +=
            chrono::duration_cast<chrono::nanoseconds>(time_end - time_start) /
            3;

        // HE encoder
        time_start = chrono::high_resolution_clock::now();
        ngraph::he::encode(input, scale, parms_id, he_plain, he_seal_backend,
                           pool);
        ngraph::he::encode(input, scale, parms_id, he_plain, he_seal_backend,
                           pool);
        ngraph::he::encode(input, scale, parms_id, he_plain, he_seal_backend,
                           pool);
        time_end = chrono::high_resolution_clock::now();
        time_he_encode_sum +=
            chrono::duration_cast<chrono::nanoseconds>(time_end - time_start);

        if (test_run == 0) {
          auto seal_plain_capacity = plain.capacity() * sizeof(std::uint64_t);
          NGRAPH_INFO << "seal::Plaintext capacity " << seal_plain_capacity;
          auto he_capacity = sizeof(std::uint64_t) * he_plain.size();
          NGRAPH_INFO << "he plaintext capacity " << he_capacity;
          NGRAPH_INFO << "Memmory improvement: "
                      << seal_plain_capacity / float(he_capacity) << "\n";
        }
      }

      // Plaintext multiplication
      if (test_run < mult_test_cnt) {
        if (coeff_modulus_bits.size() > 2) {
          Ciphertext encrypted(context);
          encryptor.encrypt(plain, encrypted);

          // SEAL
          time_start = chrono::high_resolution_clock::now();
          evaluator.multiply_plain_inplace(encrypted, plain, pool);
          time_end = chrono::high_resolution_clock::now();
          time_seal_multiply_plain_sum +=
              chrono::duration_cast<chrono::nanoseconds>(time_end - time_start);

          // HE
          time_start = chrono::high_resolution_clock::now();
          multiply_plain_inplace(encrypted, input, he_seal_backend, pool);
          time_end = chrono::high_resolution_clock::now();
          time_he_multiply_plain_sum +=
              chrono::duration_cast<chrono::nanoseconds>(time_end - time_start);
        }
      }

      // Plaintext addition
      if (test_run < add_test_cnt) {
        Ciphertext encrypted(context);
        encryptor.encrypt(plain, encrypted);

        // SEAL
        time_start = chrono::high_resolution_clock::now();
        evaluator.add_plain_inplace(encrypted, plain);
        evaluator.add_plain_inplace(encrypted, plain);
        evaluator.add_plain_inplace(encrypted, plain);
        time_end = chrono::high_resolution_clock::now();
        time_seal_add_plain_sum +=
            chrono::duration_cast<chrono::nanoseconds>(time_end - time_start) /
            3;

        // HE
        time_start = chrono::high_resolution_clock::now();
        add_plain_inplace(encrypted, input, he_seal_backend);
        add_plain_inplace(encrypted, input, he_seal_backend);
        add_plain_inplace(encrypted, input, he_seal_backend);
        time_end = chrono::high_resolution_clock::now();
        time_he_add_plain_sum +=
            chrono::duration_cast<chrono::nanoseconds>(time_end - time_start) /
            3;
      }
    }

    auto time_seal_encode_avg = time_seal_encode_sum.count() / encode_test_cnt;
    auto time_he_encode_avg = time_he_encode_sum.count() / encode_test_cnt;

    auto time_seal_multiply_plain_avg =
        time_seal_multiply_plain_sum.count() / mult_test_cnt;
    auto time_he_multiply_plain_avg = time_he_multiply_plain_sum.count() /
                                      mult_test_cnt;  //- time_he_encode_avg;

    auto time_seal_add_plain_avg =
        time_seal_add_plain_sum.count() / add_test_cnt;
    auto time_he_add_plain_avg =
        time_he_add_plain_sum.count() / add_test_cnt;  // - time_he_encode_avg;

    std::cout << "time_seal_encode_avg (ns) " << time_seal_encode_avg
              << std::endl;
    std::cout << "time_he_encode_avg (ns) " << time_he_encode_avg << std::endl;
    std::cout << "Runtime improvement: "
              << (time_seal_encode_avg / float(time_he_encode_avg)) << "\n";

    std::cout << "time_seal_multiply_plain_avg (ns) "
              << time_seal_multiply_plain_avg << std::endl;
    std::cout << "time_he_multiply_plain_avg (ns) "
              << time_he_multiply_plain_avg << std::endl;
    std::cout << "Runtime improvement: "
              << (time_seal_multiply_plain_avg /
                  float(time_he_multiply_plain_avg))
              << "\n";

    std::cout << "time_seal_add_plain_avg (ns) " << time_seal_add_plain_avg
              << std::endl;
    std::cout << "time_he_add_plain_avg (ns) " << time_he_add_plain_avg
              << std::endl;
    std::cout << "Runtime improvement: "
              << (time_seal_add_plain_avg / float(time_he_add_plain_avg))
              << "\n";
  };

  std::vector<size_t> poly_modulus_degrees{4096, 8192, 16384};

  // Bit-widths of default BFV parameters
  std::vector<std::vector<int>> coeff_modulus_bits{
      {36, 37}, {30, 30, 30, 30}, {30, 30, 30, 30, 30, 30, 30, 30, 30}};

  for (size_t parm_ind = 0; parm_ind < poly_modulus_degrees.size();
       ++parm_ind) {
    size_t poly_modulus_degree = poly_modulus_degrees[parm_ind];
    std::vector<int> coeff_moduli = coeff_modulus_bits[parm_ind];

    perf_test(poly_modulus_degree, coeff_moduli);
  }
}
