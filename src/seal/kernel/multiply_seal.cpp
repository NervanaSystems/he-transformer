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

#include "seal/kernel/multiply_seal.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/negate_seal.hpp"
#include "seal/seal_util.hpp"

void ngraph::he::scalar_multiply_seal(
    ngraph::he::SealCiphertextWrapper& arg0,
    ngraph::he::SealCiphertextWrapper& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
  if (arg0.known_value() && arg1.known_value()) {
    out->known_value() = true;
    out->value() = arg0.value() * arg1.value();
    out->complex_packing() = arg0.complex_packing();
  } else if (arg0.known_value()) {
    out->known_value() = false;
    HEPlaintext p(arg0.value());
    scalar_multiply_seal(arg1, p, out, element_type, he_seal_backend, pool);
  } else if (arg1.known_value()) {
    out->known_value() = false;
    HEPlaintext p(arg1.value());
    scalar_multiply_seal(arg0, p, out, element_type, he_seal_backend, pool);
  } else {
    NGRAPH_CHECK(arg0.complex_packing() == arg1.complex_packing(),
                 "ciphertexts must match complex form");

    match_modulus_and_scale_inplace(arg0, arg1, he_seal_backend, pool);
    size_t chain_ind0 = get_chain_index(arg0, he_seal_backend);
    size_t chain_ind1 = get_chain_index(arg1, he_seal_backend);

    if (chain_ind0 == 0 || chain_ind1 == 0) {
      NGRAPH_INFO << "Multiplicative depth limit reached";
      exit(1);
    }

    out->known_value() = false;

    if (arg0.complex_packing()) {
      seal::Ciphertext& c0 = arg0.ciphertext();
      seal::Ciphertext& c1 = arg1.ciphertext();
      seal::Ciphertext c0_conj;
      seal::Ciphertext c1_conj;

      out->complex_packing() = true;
      he_seal_backend.get_evaluator()->complex_conjugate(
          c0, *he_seal_backend.get_galois_keys(), c0_conj);
      he_seal_backend.get_evaluator()->complex_conjugate(
          c1, *he_seal_backend.get_galois_keys(), c1_conj);

      seal::Ciphertext c0_re;
      seal::Ciphertext c0_im;
      seal::Ciphertext c1_re;
      seal::Ciphertext c1_im;

      he_seal_backend.get_evaluator()->add(c0, c0_conj, c0_re);
      he_seal_backend.get_evaluator()->sub(c0, c0_conj, c0_im);
      he_seal_backend.get_evaluator()->add(c1, c1_conj, c1_re);
      he_seal_backend.get_evaluator()->sub(c1, c1_conj, c1_im);

      c0_re.scale() *= 2;
      c1_re.scale() *= 2;
      c0_im.scale() *= 2;
      c1_im.scale() *= 2;

      seal::Ciphertext prod_re;
      seal::Ciphertext prod_im;

      he_seal_backend.get_evaluator()->multiply(c0_re, c1_re, prod_re);
      he_seal_backend.get_evaluator()->multiply(c0_im, c1_im, prod_im);

      he_seal_backend.get_evaluator()->relinearize_inplace(
          prod_re, *(he_seal_backend.get_relin_keys()), pool);
      he_seal_backend.get_evaluator()->relinearize_inplace(
          prod_im, *(he_seal_backend.get_relin_keys()), pool);

      const double encode_scale = he_seal_backend.get_scale();

      auto ckks_encoder = he_seal_backend.get_ckks_encoder();
      const size_t slot_count = ckks_encoder->slot_count();
      std::vector<std::complex<double>> complex_vals(slot_count, {0, -1});
      seal::Plaintext neg_i;
      ckks_encoder->encode(complex_vals, prod_im.parms_id(), encode_scale,
                           neg_i);

      he_seal_backend.get_evaluator()->multiply_plain_inplace(prod_im, neg_i);

      std::vector<std::complex<double>> new_complex_vals(slot_count, {1, 0});
      seal::Plaintext fudge_re;
      ckks_encoder->encode(new_complex_vals, prod_re.parms_id(), encode_scale,
                           fudge_re);

      he_seal_backend.get_evaluator()->multiply_plain_inplace(prod_re,
                                                              fudge_re);
      he_seal_backend.get_evaluator()->add(prod_re, prod_im, out->ciphertext());

      he_seal_backend.get_evaluator()->rescale_to_next_inplace(
          out->ciphertext(), pool);

      out->known_value() = false;

    } else {
      NGRAPH_CHECK(arg0.complex_packing() == false,
                   "cannot multiply ciphertexts in complex form");
      NGRAPH_CHECK(arg1.complex_packing() == false,
                   "cannot multiply ciphertexts in complex form");

      if (&arg0 == &arg1) {
        he_seal_backend.get_evaluator()->square(arg0.ciphertext(),
                                                out->ciphertext(), pool);
      } else {
        he_seal_backend.get_evaluator()->multiply(
            arg0.ciphertext(), arg1.ciphertext(), out->ciphertext(), pool);
      }

      he_seal_backend.get_evaluator()->relinearize_inplace(
          out->ciphertext(), *(he_seal_backend.get_relin_keys()), pool);

      out->known_value() = false;
    }
  }
}

void ngraph::he::scalar_multiply_seal(
    ngraph::he::SealCiphertextWrapper& arg0,
    const ngraph::he::HEPlaintext& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
  if (arg0.known_value()) {
    NGRAPH_CHECK(arg1.is_single_value(), "arg1 is not single value");
    out->known_value() = true;
    out->value() = arg0.value() * arg1.first_value();
    out->complex_packing() = arg0.complex_packing();
    return;
  }
  // We can't do the scalar +/-1 optimizations, unless all the weights
  // are +/-1 in this layer, since we expect the scale of the ciphertext to
  // square. For instance, if we are computing c1*p(1) + c2 *p(2), the latter
  // sum will have larger scale than the former

  const auto& values = arg1.values();
  // TODO: check multiplying by small numbers behavior more thoroughly
  // TODO: check if abs(values) < scale?
  if (std::all_of(values.begin(), values.end(),
                  [](double f) { return std::abs(f) < 1e-5f; })) {
    out->known_value() = true;
    out->value() = 0;

  } else if (arg1.is_single_value()) {
    double value = arg1.first_value();

    multiply_plain(arg0.ciphertext(), value, out->ciphertext(), he_seal_backend,
                   pool);

    if (out->ciphertext().is_transparent()) {
      NGRAPH_WARN << "Result ciphertext is transparent";
      out->value() = 0;
    }
    out->known_value() = out->ciphertext().is_transparent();
    if (he_seal_backend.naive_rescaling()) {
      he_seal_backend.get_evaluator()->rescale_to_next_inplace(
          out->ciphertext(), pool);
    }
  } else {
    // Never complex-pack for multiplication
    auto p = SealPlaintextWrapper(false);
    ngraph::he::encode(p, arg1, *he_seal_backend.get_ckks_encoder(),
                       arg0.ciphertext().parms_id(), element_type,
                       arg0.ciphertext().scale(), false);

    size_t chain_ind0 = get_chain_index(arg0, he_seal_backend);
    size_t chain_ind1 = get_chain_index(p.plaintext(), he_seal_backend);

    NGRAPH_CHECK(chain_ind0 == chain_ind1, "Chain_ind0 ", chain_ind0,
                 " != chain_ind1 ", chain_ind1);
    NGRAPH_CHECK(chain_ind0 > 0, "Multiplicative depth exceeded for arg0");
    NGRAPH_CHECK(chain_ind1 > 0, "Multiplicative depth exceeded for arg1");

    try {
      he_seal_backend.get_evaluator()->multiply_plain(
          arg0.ciphertext(), p.plaintext(), out->ciphertext(), pool);
    } catch (const std::exception& e) {
      NGRAPH_INFO << "Error multiplying plain " << e.what();
      NGRAPH_INFO << "arg1->values().size() " << arg1.num_values();
      for (const auto& elem : arg1.values()) {
        NGRAPH_INFO << elem;
      }
    }
    out->known_value() = false;
  }
  out->complex_packing() = arg0.complex_packing();
  NGRAPH_CHECK(out->complex_packing() == he_seal_backend.complex_packing(),
               "mult out is not he_seal_backend.complex_packing()");
}

void ngraph::he::scalar_multiply_seal(const ngraph::he::HEPlaintext& arg0,
                                      const ngraph::he::HEPlaintext& arg1,
                                      ngraph::he::HEPlaintext& out,
                                      const element::Type& element_type,
                                      HESealBackend& he_seal_backend,
                                      const seal::MemoryPoolHandle& pool) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
  NGRAPH_CHECK(arg0.num_values() > 0,
               "Multiplying plaintext arg0 has 0 values");
  NGRAPH_CHECK(arg1.num_values() > 0,
               "Multiplying plaintext arg1 has 0 values");

  std::vector<double> arg0_vals = arg0.values();
  std::vector<double> arg1_vals = arg1.values();
  std::vector<double> out_vals;

  if (arg0_vals.size() == 1) {
    std::transform(arg1_vals.begin(), arg1_vals.end(),
                   std::back_inserter(out_vals),
                   std::bind(std::multiplies<double>(), std::placeholders::_1,
                             arg0_vals[0]));
  } else if (arg1_vals.size() == 1) {
    std::transform(arg0_vals.begin(), arg0_vals.end(),
                   std::back_inserter(out_vals),
                   std::bind(std::multiplies<double>(), std::placeholders::_1,
                             arg1_vals[0]));
  } else {
    NGRAPH_CHECK(arg0.num_values() == arg1.num_values(), "arg0 num values ",
                 arg0.num_values(), " != arg1 num values ", arg1.num_values(),
                 " in plain-plain multiply");

    std::transform(arg0_vals.begin(), arg0_vals.end(), arg1_vals.begin(),
                   std::back_inserter(out_vals), std::multiplies<double>());
  }
  out.set_values(out_vals);
}
