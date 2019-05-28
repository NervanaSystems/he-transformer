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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/ckks/seal_ckks_util.hpp"
#include "seal/util/polyarithsmallmod.h"

using namespace ngraph;

// Matches the modulus chain for the two elements in-place
// The elements are modified if necessary
void runtime::he::he_seal::ckks::match_modulus_and_scale_inplace(
    SealCiphertextWrapper* arg0, SealCiphertextWrapper* arg1,
    const HESealCKKSBackend* he_seal_ckks_backend,
    seal::MemoryPoolHandle pool) {
  size_t chain_ind0 =
      runtime::he::he_seal::ckks::get_chain_index(arg0, he_seal_ckks_backend);
  size_t chain_ind1 =
      runtime::he::he_seal::ckks::get_chain_index(arg1, he_seal_ckks_backend);

  if (chain_ind0 == chain_ind1) {
    return;
  }

  if (chain_ind0 < chain_ind1) {
    match_modulus_and_scale_inplace(arg1, arg0, he_seal_ckks_backend, pool);
  }

  bool rescale =
      !runtime::he::he_seal::ckks::within_rescale_tolerance(arg0, arg1);

  if (chain_ind0 > chain_ind1) {
    auto arg1_parms_id = arg1->get_hetext().parms_id();
    if (rescale) {
      he_seal_ckks_backend->get_evaluator()->rescale_to_inplace(
          arg0->get_hetext(), arg1_parms_id);
    } else {
      he_seal_ckks_backend->get_evaluator()->mod_switch_to_inplace(
          arg0->get_hetext(), arg1_parms_id);
    }
    chain_ind0 =
        runtime::he::he_seal::ckks::get_chain_index(arg0, he_seal_ckks_backend);
    NGRAPH_ASSERT(chain_ind0 == chain_ind1);

    runtime::he::he_seal::ckks::match_scale(arg0, arg1, he_seal_ckks_backend);
  }
}

// Encode value into vector of coefficients
void runtime::he::he_seal::ckks::encode(
    double value, double scale, seal::parms_id_type parms_id,
    std::vector<std::uint64_t>& destination,
    const HESealCKKSBackend* he_seal_ckks_backend,
    seal::MemoryPoolHandle pool) {
  // Verify parameters.
  auto context = he_seal_ckks_backend->get_context();
  auto context_data_ptr = context->context_data(parms_id);
  if (!context_data_ptr) {
    throw ngraph_error("parms_id is not valid for encryption parameters");
  }
  if (!pool) {
    throw ngraph_error("pool is uninitialized");
  }

  auto& context_data = *context_data_ptr;
  auto& parms = context_data.parms();
  auto& coeff_modulus = parms.coeff_modulus();
  size_t coeff_mod_count = coeff_modulus.size();
  size_t coeff_count = parms.poly_modulus_degree();

  // Quick sanity check
  if (!seal::util::product_fits_in(coeff_mod_count, coeff_count)) {
    throw ngraph_error("invalid parameters");
  }

  // Check that scale is positive and not too large
  if (scale <= 0 || (static_cast<int>(log2(scale)) >=
                     context_data.total_coeff_modulus_bit_count())) {
    throw ngraph_error("scale out of bounds");
  }

  // Compute the scaled value
  value *= scale;

  int coeff_bit_count = static_cast<int>(log2(fabs(value))) + 2;
  if (coeff_bit_count >= context_data.total_coeff_modulus_bit_count()) {
    throw ngraph_error("encoded value is too large");
  }

  double two_pow_64 = pow(2.0, 64);

  // Resize destination to appropriate size
  // TODO: use reserve?
  destination.resize(coeff_mod_count);

  double coeffd = round(value);
  bool is_negative = std::signbit(coeffd);
  coeffd = fabs(coeffd);

  NGRAPH_INFO << "Encoding double " << value << " at scale " << scale
              << " => coeffd = " << coeffd;

  // Use faster decomposition methods when possible
  if (coeff_bit_count <= 64) {
    uint64_t coeffu = static_cast<uint64_t>(fabs(coeffd));

    if (is_negative) {
      for (size_t j = 0; j < coeff_mod_count; j++) {
        destination[j] = seal::util::negate_uint_mod(
            coeffu % coeff_modulus[j].value(), coeff_modulus[j]);
        // fill_n(destination.data() + (j * coeff_count), coeff_count,
        //       negate_uint_mod(coeffu % coeff_modulus[j].value(),
        //                       coeff_modulus[j]));
      }
    } else {
      for (size_t j = 0; j < coeff_mod_count; j++) {
        destination[j] = coeffu % coeff_modulus[j].value();
        // fill_n(destination.data() + (j * coeff_count), coeff_count,
        //       coeffu % coeff_modulus[j].value());
      }
    }
  } else if (coeff_bit_count <= 128) {
    uint64_t coeffu[2]{static_cast<uint64_t>(fmod(coeffd, two_pow_64)),
                       static_cast<uint64_t>(coeffd / two_pow_64)};

    if (is_negative) {
      for (size_t j = 0; j < coeff_mod_count; j++) {
        destination[j] = seal::util::negate_uint_mod(
            seal::util::barrett_reduce_128(coeffu, coeff_modulus[j]),
            coeff_modulus[j]);
        // fill_n(destination.data() + (j * coeff_count), coeff_count,
        //       negate_uint_mod(barrett_reduce_128(coeffu, coeff_modulus[j]),
        //                       coeff_modulus[j]));
      }
    } else {
      for (size_t j = 0; j < coeff_mod_count; j++) {
        destination[j] =
            seal::util::barrett_reduce_128(coeffu, coeff_modulus[j]);
        // fill_n(destination.data() + (j * coeff_count), coeff_count,
        //        barrett_reduce_128(coeffu, coeff_modulus[j]));
      }
    }
  } else {
    // From evaluator.h
    auto decompose_single_coeff =
        [](const seal::SEALContext::ContextData& context_data,
           const std::uint64_t* value, std::uint64_t* destination,
           seal::util::MemoryPool& pool) {
          auto& parms = context_data.parms();
          auto& coeff_modulus = parms.coeff_modulus();
          std::size_t coeff_mod_count = coeff_modulus.size();
#ifdef SEAL_DEBUG
          if (value == nullptr) {
            throw ngraph_error("value cannot be null");
          }
          if (destination == nullptr) {
            throw ngraph_error("destination cannot be null");
          }
          if (destination == value) {
            throw ngraph_error("value cannot be the same as destination");
          }
#endif
          if (coeff_mod_count == 1) {
            seal::util::set_uint_uint(value, coeff_mod_count, destination);
            return;
          }

          auto value_copy(seal::util::allocate_uint(coeff_mod_count, pool));
          for (std::size_t j = 0; j < coeff_mod_count; j++) {
            // destination[j] = util::modulo_uint(
            //    value, coeff_mod_count, coeff_modulus_[j], pool);

            // Manually inlined for efficiency
            // Make a fresh copy of value
            seal::util::set_uint_uint(value, coeff_mod_count, value_copy.get());

            // Starting from the top, reduce always 128-bit blocks
            for (std::size_t k = coeff_mod_count - 1; k--;) {
              value_copy[k] = seal::util::barrett_reduce_128(
                  value_copy.get() + k, coeff_modulus[j]);
            }
            destination[j] = value_copy[0];
          }
        };

    // Slow case
    auto coeffu(seal::util::allocate_uint(coeff_mod_count, pool));
    auto decomp_coeffu(seal::util::allocate_uint(coeff_mod_count, pool));

    // We are at this point guaranteed to fit in the allocated space
    seal::util::set_zero_uint(coeff_mod_count, coeffu.get());
    auto coeffu_ptr = coeffu.get();
    while (coeffd >= 1) {
      *coeffu_ptr++ = static_cast<uint64_t>(fmod(coeffd, two_pow_64));
      coeffd /= two_pow_64;
    }

    // Next decompose this coefficient
    decompose_single_coeff(context_data, coeffu.get(), decomp_coeffu.get(),
                           pool);

    // Finally replace the sign if necessary
    if (is_negative) {
      for (size_t j = 0; j < coeff_mod_count; j++) {
        destination[j] =
            seal::util::negate_uint_mod(decomp_coeffu[j], coeff_modulus[j]);
        // fill_n(destination.data() + (j * coeff_count), coeff_count,
        //       negate_uint_mod(decomp_coeffu[j], coeff_modulus[j]));
      }
    } else {
      for (size_t j = 0; j < coeff_mod_count; j++) {
        destination[j] = decomp_coeffu[j];
        // fill_n(destination.data() + (j * coeff_count), coeff_count,
        //       decomp_coeffu[j]);
      }
    }
  }
  NGRAPH_INFO << "Encoded vals ";
  for (const auto& elem : destination) {
    NGRAPH_INFO << elem;
  }
}

void runtime::he::he_seal::ckks::add_plain_inplace(
    seal::Ciphertext& encrypted, double value,
    const HESealCKKSBackend* he_seal_ckks_backend) {
  // Verify parameters.
  auto context = he_seal_ckks_backend->get_context();
  if (!encrypted.is_metadata_valid_for(context)) {
    throw ngraph_error("encrypted is not valid for encryption parameters");
  }
  auto& context_data = *context->context_data(encrypted.parms_id());
  auto& parms = context_data.parms();

  NGRAPH_ASSERT(parms.scheme() == seal::scheme_type::CKKS)
      << "Scheme type must be CKKS";
  if (parms.scheme() == seal::scheme_type::CKKS && !encrypted.is_ntt_form()) {
    throw ngraph_error("CKKS encrypted must be in NTT form");
  }

  // Extract encryption parameters.
  auto& coeff_modulus = parms.coeff_modulus();
  size_t coeff_count = parms.poly_modulus_degree();
  size_t coeff_mod_count = coeff_modulus.size();

  // Size check
  if (!seal::util::product_fits_in(coeff_count, coeff_mod_count)) {
    throw ngraph_error("invalid parameters");
  }

  NGRAPH_INFO << "Adding double " << value << " at scale " << encrypted.scale();
  NGRAPH_INFO << "coeff_count " << coeff_count;
  NGRAPH_ASSERT(encrypted.data() != nullptr) << "Encrypted data == nullptr";

  // Encode
  std::vector<std::uint64_t> plaintext_vals(coeff_mod_count, 0);
  double scale = encrypted.scale();
  runtime::he::he_seal::ckks::encode(value, scale, parms.parms_id(),
                                     plaintext_vals, he_seal_ckks_backend);

  for (size_t j = 0; j < coeff_mod_count; j++) {
    NGRAPH_INFO << "Adding poly coeff " << j;
    // Add poly scalar instead of poly poly
    runtime::he::he_seal::ckks::add_poly_scalar_coeffmod(
        encrypted.data() + (j * coeff_count), coeff_count, plaintext_vals[j],
        coeff_modulus[j], encrypted.data() + (j * coeff_count));
    // seal::util::add_poly_poly_coeffmod(
    //     encrypted.data() + (j * coeff_count),
    //     plain.data() + (j * coeff_count), coeff_count, coeff_modulus[j],
    //     encrypted.data() + (j * coeff_count));
  }

#ifndef SEAL_ALLOW_TRANSPARENT_CIPHERTEXT
  // Transparent ciphertext output is not allowed.
  if (encrypted.is_transparent()) {
    throw ngraph_error("result ciphertext is transparent");
  }
#endif
}

void runtime::he::he_seal::ckks::multiply_plain_inplace(
    seal::Ciphertext& encrypted, double value,
    const HESealCKKSBackend* he_seal_ckks_backend,
    seal::MemoryPoolHandle pool) {
  // Verify parameters.
  auto context = he_seal_ckks_backend->get_context();
  if (!encrypted.is_metadata_valid_for(context)) {
    throw ngraph_error("encrypted is not valid for encryption parameters");
  }
  if (!context->context_data(encrypted.parms_id())) {
    throw ngraph_error("encrypted is not valid for encryption parameters");
  }
  if (!encrypted.is_ntt_form()) {
    throw ngraph_error("encrypted is not NTT form");
  }
  if (!pool) {
    throw ngraph_error("pool is uninitialized");
  }

  // Extract encryption parameters.
  auto& context_data = *context->context_data(encrypted.parms_id());
  auto& parms = context_data.parms();
  auto& coeff_modulus = parms.coeff_modulus();
  size_t coeff_count = parms.poly_modulus_degree();
  size_t coeff_mod_count = coeff_modulus.size();
  size_t encrypted_ntt_size = encrypted.size();

  // Size check
  if (!seal::util::product_fits_in(encrypted_ntt_size, coeff_count,
                                   coeff_mod_count)) {
    throw ngraph_error("invalid parameters");
  }

  std::vector<std::uint64_t> plaintext_vals(coeff_mod_count, 0);
  // TODO: explore using different scales! Smaller scales might reduce # of
  // rescalings
  double scale = encrypted.scale();
  runtime::he::he_seal::ckks::encode(value, scale, parms.parms_id(),
                                     plaintext_vals, he_seal_ckks_backend);

  double new_scale = scale * scale;
  // Check that scale is positive and not too large
  if (new_scale <= 0 || (static_cast<int>(log2(new_scale)) >=
                         context_data.total_coeff_modulus_bit_count())) {
    throw ngraph_error("scale out of bounds");
  }

  // Done doing "encoding"
  NGRAPH_INFO << "Done fake encoding " << value << " at scale " << scale;
  for (const auto& elem : plaintext_vals) {
    NGRAPH_INFO << elem;
  }

  for (size_t i = 0; i < encrypted_ntt_size; i++) {
    for (size_t j = 0; j < coeff_mod_count; j++) {
      // Multiply by scalar instead of doing dyadic product
      seal::util::multiply_poly_scalar_coeffmod(
          encrypted.data(i) + (j * coeff_count), coeff_count, plaintext_vals[j],
          coeff_modulus[j], encrypted.data(i) + (j * coeff_count));
      // dyadic_product_coeffmod(encrypted.data(i) + (j * coeff_count),
      //                        plain_ntt.data() + (j * coeff_count),
      //                        coeff_count, coeff_modulus[j],
      //                       encrypted.data(i) + (j * coeff_count));
    }
  }

  // Set the scale
  encrypted.scale() = new_scale;
  NGRAPH_INFO << "encrypted scale after mult " << encrypted.scale();

#ifndef SEAL_ALLOW_TRANSPARENT_CIPHERTEXT
  // Transparent ciphertext output is not allowed.
  if (encrypted.is_transparent()) {
    NGRAPH_INFO << "encrypted.uint64_count() " << encrypted.uint64_count();
    NGRAPH_INFO << "size " << encrypted.size();
    throw ngraph_error("result ciphertext is transparent");
  }
#endif
}
