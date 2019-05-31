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

#include <assert.h>
#include <complex>
#include <string>
#include <vector>

#include "ngraph/check.hpp"
#include "ngraph/log.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"

template <typename T>
std::string join(const T& v, const std::string& sep = ", ") {
  std::ostringstream ss;
  size_t count = 0;
  for (const auto& x : v) {
    if (count++ > 0) {
      ss << sep;
    }
    ss << x;
  }
  return ss.str();
}

static void print_seal_context(const seal::SEALContext& context) {
  auto context_data = context.context_data();
  auto scheme_parms = context_data->parms();

  assert(scheme_parms.scheme() == seal::scheme_type::CKKS);

  std::cout << "/ Encryption parameters:" << std::endl
            << "| scheme: CKKS" << std::endl
            << "| poly_modulus: " << scheme_parms.poly_modulus_degree()
            << std::endl
            // Print the size of the true (product) coefficient modulus
            << "| coeff_modulus size: "
            << context_data->total_coeff_modulus_bit_count() << " bits"
            << std::endl
            << "\\ noise_standard_deviation: "
            << scheme_parms.noise_standard_deviation() << std::endl;
}

// Packs elements of input into real values
// (a+bi, c+di) => (a,b,c,d)
auto complex_vec_to_real_vec =
    [](std::vector<double>& output,
       const std::vector<std::complex<double>>& input) {
      assert(output.size() == 0);
      for (const std::complex<double>& value : input) {
        output.emplace_back(value.real());
        output.emplace_back(value.imag());
      }
    };

// Packs elements of input into complex values
// (a,b,c,d) => (a+bi, c+di)
// (a,b,c) => (a+bi, c+0i)
auto real_vec_to_complex_vec = [](std::vector<std::complex<double>>& output,
                                  const std::vector<double>& input) {
  assert(output.size() == 0);
  std::vector<double> complex_parts(2, 0);
  for (size_t i = 0; i < input.size(); ++i) {
    complex_parts[i % 2] = input[i];

    if (i % 2 == 1 || i == input.size() - 1) {
      output.emplace_back(
          std::complex<double>(complex_parts[0], complex_parts[1]));
      complex_parts = {0, 0};
    }
  }
};

namespace ngraph {
namespace he {
template <typename S>
size_t get_chain_index(S* hetext, const HESealBackend* he_seal_backend) {
  size_t chain_ind = he_seal_backend->get_context()
                         ->context_data(hetext->get_hetext().parms_id())
                         ->chain_index();
  return chain_ind;
}

inline size_t get_chain_index(const SealPlaintextWrapper& plain,
                              const HESealBackend* he_seal_backend) {
  size_t chain_ind = he_seal_backend->get_context()
                         ->context_data(plain.m_plaintext.parms_id())
                         ->chain_index();
  return chain_ind;
}

template <typename S, typename T>
bool within_rescale_tolerance(const S* arg0, const T* arg1,
                              double factor = 1.02) {
  const auto scale0 = arg0->get_hetext().scale();
  const auto scale1 = arg1->get_hetext().scale();

  bool within_tolerance =
      (scale0 / scale1 <= factor && scale1 / scale0 <= factor);
  return within_tolerance;
}

template <typename S, typename T>
void match_scale(S* arg0, T* arg1, const HESealBackend* he_seal_backend) {
  auto scale0 = arg0->get_hetext().scale();
  auto scale1 = arg1->get_hetext().scale();
  bool scale_ok = within_rescale_tolerance(arg0, arg1);
  NGRAPH_CHECK(scale_ok, "Scale ", scale0, "does not match scale ", scale1,
               " in scalar add");
  arg0->get_hetext().scale() = arg1->get_hetext().scale();
}

void match_modulus_and_scale_inplace(
    SealCiphertextWrapper* arg0, SealCiphertextWrapper* arg1,
    const HESealBackend* he_seal_backend,
    seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());

void encode(double value, double scale, seal::parms_id_type parms_id,
            std::vector<std::uint64_t>& destination,
            const HESealBackend* he_seal_backend,
            seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());

void add_plain_inplace(seal::Ciphertext& encrypted, double value,
                       const HESealBackend* he_seal_backend);

inline void add_plain(const seal::Ciphertext& encrypted, double value,
                      seal::Ciphertext& destination,
                      const HESealBackend* he_seal_backend) {
  destination = encrypted;
  ngraph::he::add_plain_inplace(destination, value, he_seal_backend);
}

// Like add_poly_poly_coeffmod, but with a scalar for operand2
inline void add_poly_scalar_coeffmod(const std::uint64_t* poly,
                                     std::size_t coeff_count,
                                     std::uint64_t scalar,
                                     const seal::SmallModulus& modulus,
                                     std::uint64_t* result) {
  const uint64_t modulus_value = modulus.value();
#ifdef SEAL_DEBUG
  if (poly == nullptr && coeff_count > 0) {
    throw ngraph_error("poly");
  }
  if (scalar >= modulus_value) {
    throw ngraph_error("scalar");
  }
  if (modulus.is_zero()) {
    throw ngraph_error("modulus");
  }
  if (result == nullptr && coeff_count > 0) {
    throw ngraph_error("result");
  }
#endif

  for (; coeff_count--; result++, poly++) {
    // Explicit inline
    // result[i] = add_uint_uint_mod(poly[i], scalar, modulus);
#ifdef SEAL_DEBUG
    if (*poly >= modulus_value) {
      throw ngraph_error("poly > modulus_value");
    }

#endif
    std::uint64_t sum = *poly + scalar;
    *result = sum - (modulus_value &
                     static_cast<std::uint64_t>(
                         -static_cast<std::int64_t>(sum >= modulus_value)));
  }
}

void multiply_plain_inplace(
    seal::Ciphertext& encrypted, double value,
    const HESealBackend* he_seal_backend,
    seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());

inline void multiply_plain(
    const seal::Ciphertext& encrypted, double value,
    seal::Ciphertext& destination, const HESealBackend* he_seal_backend,
    seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {
  destination = encrypted;
  ngraph::he::multiply_plain_inplace(destination, value, he_seal_backend,
                                     std::move(pool));
}
}  // namespace he
}  // namespace ngraph
