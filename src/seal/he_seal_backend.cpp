/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <limits>

#include "he_cipher_tensor.hpp"
#include "he_plain_tensor.hpp"
#include "he_seal_backend.hpp"
#include "he_seal_parameter.hpp"
#include "he_tensor.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/bfv/he_seal_bfv_backend.hpp"

#include "seal/seal.h"

using namespace ngraph;
using namespace std;


extern "C" const char* get_ngraph_version_string()
{
    return "v0.9.0"; // TODO: move to CMakeLists
}

extern "C" runtime::Backend* new_backend(const char* configuration_chars)
{
    string configuration_string = string(configuration_chars);

    if (configuration_string == "HE:SEAL:BFV")
    {
        return new runtime::he::he_seal::HESealBFVBackend();
    }
    else if (configuration_string == "HE:SEAL:CKKS")
    {
       return new runtime::he::he_seal::HESealCKKSBackend();
    }
    else
    {
        throw ngraph_error("Invalid configuration string \"" + configuration_string + "\" in new_backend");
    }
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}


void runtime::he::he_seal::HESealBackend::assert_valid_seal_parameter(const shared_ptr<runtime::he::he_seal::HESealParameter> sp) const
{
    if (sp->m_scheme_name != "HE:SEAL:BFV" && sp->m_scheme_name != "HE:SEAL:CKKS")
    {
        throw ngraph_error("Invalid scheme name");
    }
    static unordered_set<uint64_t> valid_poly_modulus{1024, 2048, 4096, 8192, 16384, 32768};
    if (valid_poly_modulus.count(sp->m_poly_modulus_degree) == 0)
    {
        throw ngraph_error("m_poly_modulus must be 1024, 2048, 4096, 8192, 16384, 32768");
    }

    if (sp->m_security_level != 128 && sp->m_security_level != 192)
    {
        throw ngraph_error("sp.security_level must be 128, 192");
    }

    if (sp->m_evaluation_decomposition_bit_count > 60 || sp->m_evaluation_decomposition_bit_count < 1)
    {
        throw ngraph_error("sp.m_evaluation_decomposition_bit_count must be between 1 and 60");
    }
}

/* runtime::he::he_seal::HESealBackend::HESealBackend(
    const shared_ptr<runtime::he::he_seal::HESealParameter>& sp)
{
    assert_valid_seal_parameter(sp);
    NGRAPH_INFO << "Making HESealBackend " << sp->m_scheme_name;
    // Context
    m_context = make_seal_context(sp);
    auto m_context_data = m_context->context_data();
    print_seal_context(*m_context);

    // Encoders
    auto poly_modulus = m_context_data->parms().plain_modulus().value();
    auto plain_modulus = m_context_data->parms().plain_modulus().value();

    // Keygen, encryptor and decryptor
    m_keygen = make_shared<seal::KeyGenerator>(m_context);
    m_relin_key = make_shared<seal::RelinKeys>(m_keygen->relin_keys(16));
    m_public_key = make_shared<seal::PublicKey>(m_keygen->public_key());
    m_secret_key = make_shared<seal::SecretKey>(m_keygen->secret_key());
    m_encryptor = make_shared<seal::Encryptor>(m_context, *m_public_key);
    m_decryptor = make_shared<seal::Decryptor>(m_context, *m_secret_key);

    // Evaluator
    m_evaluator = make_shared<seal::Evaluator>(m_context);
} */

/*
shared_ptr<seal::SEALContext> runtime::he::he_seal::HESealBackend::make_seal_context(
    const shared_ptr<runtime::he::he_seal::HESealParameter> sp) const
{
    seal::EncryptionParameters parms = (sp->m_scheme_name == "HE:SEAL:BFV" ? seal::scheme_type::BFV :
                                        sp->m_scheme_name == "HE:SEAL:CKKS" ? seal::scheme_type::CKKS :
                                        throw ngraph_error("Invalid scheme name \"" + sp->m_scheme_name + "\""));

    NGRAPH_INFO << "Using CKKS scheme? " << (parms == seal::scheme_type::CKKS);

    NGRAPH_INFO << "Setting poly mod degree to " << sp->m_poly_modulus_degree;

    parms.set_poly_modulus_degree(sp->m_poly_modulus_degree);

    NGRAPH_INFO << "Setting coeff mod to security level " << sp->m_security_level;
    if (sp->m_security_level == 128)
    {
        parms.set_coeff_modulus(seal::coeff_modulus_128(sp->m_poly_modulus_degree));
    }
    else if (sp->m_security_level == 192)
    {
        parms.set_coeff_modulus(seal::coeff_modulus_192(sp->m_poly_modulus_degree));
    }
    else
    {
        throw ngraph_error("sp.security_level must be 128, 192");
    }


    auto tmp =seal::SEALContext::Create(parms);

    NGRAPH_INFO << "Created SEALContext(parmz)";

    return  tmp;
}
*/

shared_ptr<runtime::he::HECiphertext>
    runtime::he::he_seal::HESealBackend::create_empty_ciphertext() const
{
    return make_shared<runtime::he::he_seal::SealCiphertextWrapper>();
}

shared_ptr<runtime::he::HEPlaintext>
    runtime::he::he_seal::HESealBackend::create_empty_plaintext() const
{
    return make_shared<SealPlaintextWrapper>();
}

void runtime::he::he_seal::HESealBackend::encrypt(
    shared_ptr<runtime::he::HECiphertext>& output,
    const shared_ptr<runtime::he::HEPlaintext> input) const
{
    auto seal_output = dynamic_pointer_cast<runtime::he::he_seal::SealCiphertextWrapper>(output);
    auto seal_input = dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(input);
    if (seal_output != nullptr && seal_input != nullptr)
    {
        m_encryptor->encrypt(seal_input->m_plaintext, seal_output->m_ciphertext);
    }
    else
    {
        throw ngraph_error("HESealBackend::encrypt has non-seal ciphertexts");
    }
}

void runtime::he::he_seal::HESealBackend::decrypt(
    shared_ptr<runtime::he::HEPlaintext>& output,
    const shared_ptr<runtime::he::HECiphertext> input) const
{
    auto seal_output = dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(output);
    auto seal_input = dynamic_pointer_cast<runtime::he::he_seal::SealCiphertextWrapper>(input);
    if (seal_output != nullptr && seal_input != nullptr)
    {
        m_decryptor->decrypt(seal_input->m_ciphertext, seal_output->m_plaintext);
    }
    else
    {
        throw ngraph_error("HESealBackend::decrypt has non-seal ciphertexts");
    }
}

const shared_ptr<runtime::he::HEPlaintext>
    runtime::he::he_seal::HESealBackend::get_valued_plaintext(int64_t value) const
{
    std::unordered_set<int64_t> stored_plaintext_values{-1, 0, 1};
    if (stored_plaintext_values.find(value) == stored_plaintext_values.end())
    {
        throw ngraph_error("Value not stored in stored plaintext values");
    }
    auto plaintext = create_empty_plaintext();
    plaintext = m_plaintext_map.at(value);
    return plaintext;
    // return m_plaintext_map.at(value);
}