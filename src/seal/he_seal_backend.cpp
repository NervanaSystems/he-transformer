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

#include "seal/seal.h"

using namespace ngraph;
using namespace std;


static void print_seal_context(const seal::SEALContext& context)
{
    auto context_data = context.context_data();
    NGRAPH_INFO << endl
                << "/ Encryption parameters:" << endl
                << "| poly_modulus: " << context_data->parms().poly_modulus_degree() << endl
                // Print the size of the true (product) coefficient modulus
                << "| coeff_modulus size: " << context_data->total_coeff_modulus().significant_bit_count()
                << " bits" << endl
                << "| plain_modulus: " << context_data->parms().plain_modulus().value() << endl
                << "\\ noise_standard_deviation: " << context_data->parms().noise_standard_deviation();
}

extern "C" const char* get_ngraph_version_string()
{
    return "v0.9.0"; // TODO: move to CMakeLists
}

runtime::he::he_seal::HESealBackend::HESealBackend(
    const shared_ptr<runtime::he::he_seal::HESealParameter>& sp)
{
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
}

shared_ptr<seal::SEALContext> runtime::he::he_seal::HESealBackend::make_seal_context(
    const shared_ptr<runtime::he::he_seal::HESealParameter> sp) const
{
    seal::EncryptionParameters parms = (sp->m_scheme_name == "BFV" ? seal::scheme_type::BFV :
                                        seal::scheme_type::CKKS);
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

    NGRAPH_INFO << "Setting plain mod to " << sp->m_plain_modulus;
    parms.set_plain_modulus(sp->m_plain_modulus);

    return seal::SEALContext::Create(parms);
}

shared_ptr<runtime::Tensor>
    runtime::he::he_seal::HESealBackend::create_tensor(const element::Type& element_type,
                                                       const Shape& shape)
{
    auto rc = make_shared<runtime::he::HECipherTensor>(element_type, shape, create_empty_ciphertext());
    return static_pointer_cast<runtime::Tensor>(rc);
}

shared_ptr<runtime::Tensor>
    runtime::he::he_seal::HESealBackend::create_plain_tensor(const element::Type& element_type,
                                                             const Shape& shape)
{
    auto rc = make_shared<runtime::he::HEPlainTensor>(element_type, shape, create_empty_plaintext());
    return static_pointer_cast<runtime::Tensor>(rc);
}


shared_ptr<runtime::he::HECiphertext>
    runtime::he::he_seal::HESealBackend::create_empty_ciphertext() const
{
    return make_shared<runtime::he::he_seal::SealCiphertextWrapper>();
}

shared_ptr<runtime::Tensor> runtime::he::he_seal::HESealBackend::create_valued_plain_tensor(
    float value, const element::Type& element_type, const Shape& shape)
{
    auto tensor = static_pointer_cast<HEPlainTensor>(create_plain_tensor(element_type, shape));
    vector<shared_ptr<runtime::he::HEPlaintext>>& plain_texts = tensor->get_elements();
#pragma omp parallel for
    for (size_t i = 0; i < plain_texts.size(); ++i)
    {
        plain_texts[i] = create_valued_plaintext(value, element_type);
    }
    return tensor;
}

/* shared_ptr<runtime::he::HEPlaintext> runtime::he::he_seal::HESealBackend::create_valued_plaintext(
    float value, const element::Type& element_type) const
{
    const string type_name = element_type.c_type_string();
    shared_ptr<runtime::he::HEPlaintext> plaintext = create_empty_plaintext();
    if (auto plaintext_seal = dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(plaintext))
    {
        if (type_name == "float")
        {
            plaintext_seal->m_plaintext = m_frac_encoder->encode(value);
        }
        else if (type_name == "int64_t")
        {
            plaintext_seal->m_plaintext = m_int_encoder->encode(static_cast<int64_t>(value));
        }
        else
        {
            throw ngraph_error("Type not supported at create_valued_plaintext");
        }
    }
    else
    {
        NGRAPH_INFO << "Plaintext is not SEAL type in create_valued_plaintext";
    }
    return plaintext;
}

shared_ptr<runtime::he::HEPlaintext>
    runtime::he::he_seal::HESealBackend::get_valued_plaintext(int64_t value,
                                                              const element::Type& element_type)
{
    const string type_name = element_type.c_type_string();
    std::unordered_set<int64_t> stored_plaintext_values{-1, 0, 1};
    if (stored_plaintext_values.find(value) == stored_plaintext_values.end())
    {
        throw ngraph_error("Value not stored in stored plaintext values");
    }
    if ((m_plaintext_map.find(type_name) == m_plaintext_map.end()) ||
        m_plaintext_map[type_name].find(value) == m_plaintext_map[type_name].end())
    {
        throw ngraph_error("Type or value not stored in m_plaintext_map");
    }
    return m_plaintext_map[type_name][value];
 } */

shared_ptr<runtime::he::HEPlaintext>
    runtime::he::he_seal::HESealBackend::create_empty_plaintext() const
{
    return make_shared<SealPlaintextWrapper>();
}

shared_ptr<runtime::Tensor> runtime::he::he_seal::HESealBackend::create_valued_tensor(
    float value, const element::Type& element_type, const Shape& shape)
{
    auto tensor = static_pointer_cast<HECipherTensor>(create_tensor(element_type, shape));
    vector<shared_ptr<runtime::he::HECiphertext>>& cipher_texts = tensor->get_elements();
#pragma omp parallel for
    for (size_t i = 0; i < cipher_texts.size(); ++i)
    {
        cipher_texts[i] = create_valued_ciphertext(value, element_type);
    }
    return tensor;
}
/*
shared_ptr<runtime::Tensor> runtime::he::he_seal::HESealBackend::create_valued_plain_tensor(
    float value, const element::Type& element_type, const Shape& shape)
{
    auto tensor = static_pointer_cast<HEPlainTensor>(create_plain_tensor(element_type, shape));
    vector<shared_ptr<runtime::he::HEPlaintext>>& plain_texts = tensor->get_elements();
#pragma omp parallel for
    for (size_t i = 0; i < plain_texts.size(); ++i)
    {
        plain_texts[i] = create_valued_plaintext(value, element_type);
    }
    return tensor;
} */

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
    m_decryptor->decrypt(seal_input->m_ciphertext, seal_output->m_plaintext);
}