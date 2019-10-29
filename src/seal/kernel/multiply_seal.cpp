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

namespace ngraph {
namespace he {

void scalar_multiply_seal(SealCiphertextWrapper& arg0,
                          SealCiphertextWrapper& arg1,
                          std::shared_ptr<SealCiphertextWrapper>& out,
                          bool complex_packing, HESealBackend& he_seal_backend,
                          const seal::MemoryPoolHandle& pool) {
  match_modulus_and_scale_inplace(arg0, arg1, he_seal_backend, pool);
  size_t chain_ind0 = he_seal_backend.get_chain_index(arg0);
  size_t chain_ind1 = he_seal_backend.get_chain_index(arg1);

  if (chain_ind0 == 0 || chain_ind1 == 0) {
    NGRAPH_ERR << "Multiplicative depth limit reached";
    throw ngraph_error("Multiplicative depth reached");
  }

  if (complex_packing) {
    // Compute c0 x c1 == ((c0 - c0*)(c1 - c1*) + (-i)(c0 + c0*)(c1 + c1*))/4

    seal::Ciphertext& c0 = arg0.ciphertext();
    seal::Ciphertext& c1 = arg1.ciphertext();
    seal::Ciphertext c0_conj, c1_conj;

    he_seal_backend.get_evaluator()->complex_conjugate(
        c0, *he_seal_backend.get_galois_keys(), c0_conj);
    he_seal_backend.get_evaluator()->complex_conjugate(
        c1, *he_seal_backend.get_galois_keys(), c1_conj);

    seal::Ciphertext c0_re, c0_im, c1_re, c1_im;

    he_seal_backend.get_evaluator()->add(c0, c0_conj, c0_re);
    he_seal_backend.get_evaluator()->sub(c0, c0_conj, c0_im);
    he_seal_backend.get_evaluator()->add(c1, c1_conj, c1_re);
    he_seal_backend.get_evaluator()->sub(c1, c1_conj, c1_im);

    // Divide by two, since (a+bi) + (a+bi)* = 2a, etc.
    c0_re.scale() *= 2;
    c1_re.scale() *= 2;
    c0_im.scale() *= 2;
    c1_im.scale() *= 2;

    seal::Ciphertext prod_re, prod_im;

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
    ckks_encoder->encode(complex_vals, prod_im.parms_id(), encode_scale, neg_i);

    he_seal_backend.get_evaluator()->multiply_plain_inplace(prod_im, neg_i);

    std::vector<std::complex<double>> new_complex_vals(slot_count, {1, 0});
    seal::Plaintext fudge_re;
    ckks_encoder->encode(new_complex_vals, prod_re.parms_id(), encode_scale,
                         fudge_re);

    he_seal_backend.get_evaluator()->multiply_plain_inplace(prod_re, fudge_re);
    he_seal_backend.get_evaluator()->add(prod_re, prod_im, out->ciphertext());

    he_seal_backend.get_evaluator()->rescale_to_next_inplace(out->ciphertext(),
                                                             pool);
  } else {
    if (&arg0 == &arg1) {
      he_seal_backend.get_evaluator()->square(arg0.ciphertext(),
                                              out->ciphertext(), pool);
    } else {
      he_seal_backend.get_evaluator()->multiply(
          arg0.ciphertext(), arg1.ciphertext(), out->ciphertext(), pool);
    }

    he_seal_backend.get_evaluator()->relinearize_inplace(
        out->ciphertext(), *(he_seal_backend.get_relin_keys()), pool);
  }
}

void scalar_multiply_seal(SealCiphertextWrapper& arg0, const HEPlaintext& arg1,
                          HEType& out, HESealBackend& he_seal_backend,
                          const seal::MemoryPoolHandle& pool) {
  // TODO(fboemer): check multiplying by small numbers behavior more thoroughly
  // TODO(fboemer): check if abs(values) < scale?
  if (std::all_of(arg1.begin(), arg1.end(),
                  [](double f) { return std::abs(f) < 1e-5f; })) {
    HEPlaintext zeros({std::vector<double>(arg1.size(), 0)});
    out.set_plaintext(zeros);
  } else if (arg1.size() == 1) {
    if (!out.is_ciphertext()) {
      auto empty_cipher = HESealBackend::create_empty_ciphertext();
      out.set_ciphertext(empty_cipher);
    }
    multiply_plain(arg0.ciphertext(), arg1[0],
                   out.get_ciphertext()->ciphertext(), he_seal_backend, pool);

    if (out.get_ciphertext()->ciphertext().is_transparent()) {
      HEPlaintext zeros({std::vector<double>(arg1.size(), 0)});
      out.set_plaintext(zeros);
    } else if (he_seal_backend.naive_rescaling()) {
      he_seal_backend.get_evaluator()->rescale_to_next_inplace(
          out.get_ciphertext()->ciphertext(), pool);
    }
  } else {
    if (!out.is_ciphertext()) {
      auto empty_cipher = HESealBackend::create_empty_ciphertext();
      out.set_ciphertext(empty_cipher);
    }

    // Never complex-pack for multiplication
    auto p = SealPlaintextWrapper(false);
    encode(p, arg1, *he_seal_backend.get_ckks_encoder(),
           arg0.ciphertext().parms_id(), element::f32,
           arg0.ciphertext().scale(), false);

    size_t chain_ind0 = he_seal_backend.get_chain_index(arg0);
    size_t chain_ind1 = he_seal_backend.get_chain_index(p.plaintext());

    NGRAPH_CHECK(chain_ind0 == chain_ind1, "Chain_ind0 ", chain_ind0,
                 " != chain_ind1 ", chain_ind1);
    NGRAPH_CHECK(chain_ind0 > 0, "Multiplicative depth exceeded for arg0");
    NGRAPH_CHECK(chain_ind1 > 0, "Multiplicative depth exceeded for arg1");

    try {
      he_seal_backend.get_evaluator()->multiply_plain(
          arg0.ciphertext(), p.plaintext(), out.get_ciphertext()->ciphertext(),
          pool);
    } catch (const std::exception& e) {
      NGRAPH_ERR << "Error multiplying plain " << e.what();
      NGRAPH_ERR << "arg1->values().size() " << arg1.size();
      NGRAPH_ERR << "arg1 " << arg1;
      throw ngraph_error("Error multiplying plain");
    }
  }
}

void scalar_multiply_seal(const HEPlaintext& arg0, const HEPlaintext& arg1,
                          HEPlaintext& out) {
  HEPlaintext out_vals;
  if (arg0.size() == 1) {
    std::transform(
        arg1.begin(), arg1.end(), std::back_inserter(out_vals),
        std::bind(std::multiplies<>(), std::placeholders::_1, arg0[0]));
  } else if (arg1.size() == 1) {
    std::transform(
        arg0.begin(), arg0.end(), std::back_inserter(out_vals),
        std::bind(std::multiplies<>(), std::placeholders::_1, arg1[0]));
  } else {
    NGRAPH_CHECK(arg0.size() == arg1.size(), "arg0.size() ", arg0.size(),
                 " != arg0.size() ", arg1.size(), " in plain-plain multiply");
    std::transform(arg0.begin(), arg0.end(), arg1.begin(),
                   std::back_inserter(out_vals), std::multiplies<>());
  }
  out = std::move(out_vals);
}

}  // namespace he
}  // namespace ngraph
