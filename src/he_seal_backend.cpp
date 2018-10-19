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

#include "he_call_frame.hpp"
#include "he_cipher_tensor_view.hpp"
#include "he_plain_tensor_view.hpp"
#include "he_seal_backend.hpp"
#include "he_seal_parameter.hpp"
#include "he_tensor_view.hpp"

#include "seal/seal.h"

using namespace ngraph;
using namespace std;

const static runtime::he::HESealParameter parse_seal_config_or_use_default()
{
    try
    {
        const char* config_path = std::getenv("NGRAPH_HE_SEAL_CONFIG");
        if (config_path != nullptr)
        {
            // Read file to string
            std::ifstream f(config_path);
            std::stringstream ss;
            ss << f.rdbuf();
            std::string s = ss.str();

            // Parse json
            nlohmann::json js = nlohmann::json::parse(s);
            std::uint64_t poly_modulus = js["poly_modulus"];
            std::uint64_t plain_modulus = js["plain_modulus"];
            std::uint64_t security_level = js["security_level"];
            int fractional_encoder_integer_coeff_count =
                js["fractional_encoder_integer_coeff_count"];
            int fractional_encoder_fraction_coeff_count =
                js["fractional_encoder_fraction_coeff_count"];
            std::uint64_t fractional_encoder_base = js["fractional_encoder_base"];
            int evaluation_decomposition_bit_count = js["evaluation_decomposition_bit_count"];

            NGRAPH_INFO << "Using SEAL config for parameters: " << config_path;
            return runtime::he::HESealParameter(poly_modulus,
                                                plain_modulus,
                                                security_level,
                                                fractional_encoder_integer_coeff_count,
                                                fractional_encoder_fraction_coeff_count,
                                                fractional_encoder_base,
                                                evaluation_decomposition_bit_count);
        }
        else
        {
            NGRAPH_INFO << "Using SEAL default parameters" << config_path;
            throw std::runtime_error("config_path is NULL");
        }
    }
    catch (const std::exception& e)
    {
        return runtime::he::HESealParameter(8192,      // poly_modulus
                                            2L << 30L, // plain_modulus
                                            128,       // security_level
                                            64,        // fractional_encoder_integer_coeff_count
                                            32,        // fractional_encoder_fraction_coeff_count
                                            2,         // fractional_encoder_base
                                            16         // evaluation_decomposition_bit_count
                                            );
    }
}

const static runtime::he::HESealParameter default_seal_parameter =
    parse_seal_config_or_use_default();

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

runtime::he::he_seal::HESealBackend::HESealBackend()
    : runtime::he::he_seal::HESealBackend(
          make_shared<runtime::he::HESealParameter>(default_seal_parameter))
{
}

runtime::he::he_seal::HESealBackend::HESealBackend(
    const shared_ptr<runtime::he::HESealParameter> sp)
{
    assert_valid_seal_parameter(sp);

    // Context
    m_context = make_seal_context(sp);
    auto m_context_data = m_context->context_data();
    print_seal_context(*m_context);

    // Encoders
    auto poly_modulus = m_context_data->parms().plain_modulus().value();
    auto plain_modulus = m_context_data->parms().plain_modulus().value();

    m_int_encoder = make_shared<seal::IntegerEncoder>(plain_modulus);
    m_frac_encoder =
        make_shared<seal::FractionalEncoder>(plain_modulus,
                                             poly_modulus,
                                             sp->m_fractional_encoder_integer_coeff_count,
                                             sp->m_fractional_encoder_fraction_coeff_count,
                                             sp->m_fractional_encoder_base);

    // Keygen, encryptor and decryptor
    m_keygen = make_shared<seal::KeyGenerator>(m_context);
    m_relin_key = make_shared<seal::RelinKeys>(m_keygen->relin_keys(16));
    m_public_key = make_shared<seal::PublicKey>(m_keygen->public_key());
    m_secret_key = make_shared<seal::SecretKey>(m_keygen->secret_key());
    m_encryptor = make_shared<seal::Encryptor>(m_context, *m_public_key);
    m_decryptor = make_shared<seal::Decryptor>(m_context, *m_secret_key);

    // Evaluator
    m_evaluator = make_shared<seal::Evaluator>(m_context);

    // Plaintext constants
    m_plaintext_map["float"][0] = make_shared<SealPlaintextWrapper>(m_frac_encoder->encode(0));
    m_plaintext_map["float"][1] = make_shared<SealPlaintextWrapper>(m_frac_encoder->encode(1));
    m_plaintext_map["float"][-1] = make_shared<SealPlaintextWrapper>(m_frac_encoder->encode(-1));
    m_plaintext_map["int64_t"][0] = make_shared<SealPlaintextWrapper>(m_int_encoder->encode(0));
    m_plaintext_map["int64_t"][1] = make_shared<SealPlaintextWrapper>(m_int_encoder->encode(1));
    m_plaintext_map["int64_t"][-1] = make_shared<SealPlaintextWrapper>(m_int_encoder->encode(-1));
}

runtime::he::he_seal::HESealBackend::~HESealBackend()
{
}

void runtime::he::he_seal::HESealBackend::assert_valid_seal_parameter(
    const shared_ptr<runtime::he::HESealParameter> sp) const
{
    static unordered_set<uint64_t> valid_poly_modulus{1024, 2048, 4096, 8192, 16384, 32768};
    if (valid_poly_modulus.count(sp->m_poly_modulus) == 0)
    {
        throw ngraph_error("m_poly_modulus must be 1024, 2048, 4096, 8192, 16384, 32768");
    }
    if (sp->m_security_level != 128 && sp->m_security_level != 192)
    {
        throw ngraph_error("sp.security_level must be 128, 192");
    }
}

shared_ptr<seal::SEALContext> runtime::he::he_seal::HESealBackend::make_seal_context(
    const shared_ptr<runtime::he::HESealParameter> sp) const
{
    assert_valid_seal_parameter(sp);

    seal::EncryptionParameters parms(seal::scheme_type::BFV);
    parms.set_poly_modulus_degree(sp->m_poly_modulus);
    if (sp->m_security_level == 128)
    {
        parms.set_coeff_modulus(seal::coeff_modulus_128(sp->m_poly_modulus));
    }
    else if (sp->m_security_level == 192)
    {
        parms.set_coeff_modulus(seal::coeff_modulus_192(sp->m_poly_modulus));
    }
    else
    {
        throw ngraph_error("sp.security_level must be 128, 192");
    }
    parms.set_plain_modulus(sp->m_plain_modulus);

    return seal::SEALContext::Create(parms);
}

shared_ptr<runtime::TensorView>
    runtime::he::he_seal::HESealBackend::create_tensor(const element::Type& element_type,
                                                       const Shape& shape)
{
    shared_ptr<HESealBackend> he_seal_backend =
        dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(shared_from_this());
    auto rc = make_shared<runtime::he::HECipherTensorView>(element_type, shape, he_seal_backend);
    return static_pointer_cast<runtime::TensorView>(rc);
}

shared_ptr<runtime::TensorView> runtime::he::he_seal::HESealBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, const bool batched)
{
    if (batched)
    {
        throw ngraph_error("HESealBackend does not support batched create tensor");
    }
    else
    {
        create_tensor(element_type, shape);
    }
}

shared_ptr<runtime::TensorView> runtime::he::he_seal::HESealBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    throw ngraph_error("HESeal create_tensor unimplemented");
}

shared_ptr<runtime::TensorView>
    runtime::he::he_seal::HESealBackend::create_plain_tensor(const element::Type& element_type,
                                                             const Shape& shape)
{
    shared_ptr<HESealBackend> he_seal_backend =
        dynamic_pointer_cast<runtime::he::he_seal::HESealBackend>(shared_from_this());
    auto rc = make_shared<runtime::he::HEPlainTensorView>(element_type, shape, he_seal_backend);
    return static_pointer_cast<runtime::TensorView>(rc);
}

shared_ptr<runtime::he::HECiphertext> runtime::he::he_seal::HESealBackend::create_valued_ciphertext(
    float value, const element::Type& element_type, const seal::MemoryPoolHandle& pool) const
{
    throw ngraph_error("create_valued_ciphertext unimplemented");
}

shared_ptr<runtime::he::HECiphertext> runtime::he::he_seal::HESealBackend::create_empty_ciphertext(
    const seal::MemoryPoolHandle& pool) const
{
    throw ngraph_error("HESealBackend::create_empty_ciphertext unimplemented");
}

shared_ptr<runtime::he::HEPlaintext> runtime::he::he_seal::HESealBackend::create_valued_plaintext(
    float value, const element::Type& element_type, const seal::MemoryPoolHandle& pool) const
{
    throw ngraph_error("HESealBackend::create_empty_plainttext unimplemented");
}

shared_ptr<runtime::he::HEPlaintext> runtime::he::he_seal::HESealBackend::create_empty_plaintext(
    const seal::MemoryPoolHandle& pool) const
{
    throw ngraph_error("HESealBackend::create_empty_plaintext unimplemnented");
}

shared_ptr<runtime::he::HECiphertext> runtime::he::he_seal::HESealBackend::create_valued_ciphertext(
    float value, const element::Type& element_type, size_t batch_size) const
{
    if (batch_size != 1)
    {
        throw ngraph_error("HESealBackend::create_valued_ciphertext only supports batch size 1");
    }
    const string type_name = element_type.c_type_string();
    auto ciphertext =
        dynamic_pointer_cast<runtime::he::SealCiphertextWrapper>(create_empty_ciphertext());
    if (ciphertext == nullptr)
    {
        throw ngraph_error("Ciphertext is not seal ciphertext in create_valued_ciphertext");
    }
    if (type_name == "float")
    {
        seal::Plaintext plaintext = m_frac_encoder->encode(value);
        m_encryptor->encrypt(plaintext, ciphertext->m_ciphertext);
    }
    else if (type_name == "int64_t")
    {
        seal::Plaintext plaintext = m_int_encoder->encode(static_cast<int64_t>(value));
        m_encryptor->encrypt(plaintext, ciphertext->m_ciphertext);
    }
    else
    {
        throw ngraph_error("Type not supported at create_ciphertext");
    }
    return ciphertext;
}

shared_ptr<runtime::he::HECiphertext>
    runtime::he::he_seal::HESealBackend::create_empty_ciphertext(size_t batch_size) const
{
    if (batch_size != 1)
    {
        throw ngraph_error("HESealBackend::create_empty_ciphertext only supports batch size 1");
    }
    return make_shared<runtime::he::SealCiphertextWrapper>();
}

shared_ptr<runtime::he::HEPlaintext> runtime::he::he_seal::HESealBackend::create_valued_plaintext(
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
}

shared_ptr<runtime::he::HEPlaintext>
    runtime::he::he_seal::HESealBackend::create_empty_plaintext() const
{
    return make_shared<SealPlaintextWrapper>();
}

shared_ptr<runtime::TensorView> runtime::he::he_seal::HESealBackend::create_valued_tensor(
    float value, const element::Type& element_type, const Shape& shape)
{
    auto tensor = static_pointer_cast<HECipherTensorView>(create_tensor(element_type, shape));
    vector<shared_ptr<runtime::he::HECiphertext>>& cipher_texts = tensor->get_elements();
#pragma omp parallel for
    for (size_t i = 0; i < cipher_texts.size(); ++i)
    {
        cipher_texts[i] = create_valued_ciphertext(value, element_type);
    }
    return tensor;
}

shared_ptr<runtime::TensorView> runtime::he::he_seal::HESealBackend::create_valued_plain_tensor(
    float value, const element::Type& element_type, const Shape& shape)
{
    auto tensor = static_pointer_cast<HEPlainTensorView>(create_plain_tensor(element_type, shape));
    vector<shared_ptr<runtime::he::HEPlaintext>>& plain_texts = tensor->get_elements();
#pragma omp parallel for
    for (size_t i = 0; i < plain_texts.size(); ++i)
    {
        plain_texts[i] = create_valued_plaintext(value, element_type);
    }
    return tensor;
}

void runtime::he::he_seal::HESealBackend::encode(shared_ptr<runtime::he::HEPlaintext>& output,
                                                 const void* input,
                                                 const element::Type& type,
                                                 size_t count) const
{
    if (count != 1)
    {
        throw ngraph_error("Batching not enabled for SEAL in encode");
    }
    const string type_name = type.c_type_string();

    if (type_name == "int64_t")
    {
        output =
            make_shared<runtime::he::SealPlaintextWrapper>(m_int_encoder->encode(*(int64_t*)input));
    }
    else if (type_name == "float")
    {
        output =
            make_shared<runtime::he::SealPlaintextWrapper>(m_frac_encoder->encode(*(float*)input));
    }
    else
    {
        NGRAPH_INFO << "Unsupported element type in decode " << type_name;
        throw ngraph_error("Unsupported element type " + type_name);
    }
}

void runtime::he::he_seal::HESealBackend::decode(void* output,
                                                 const shared_ptr<runtime::he::HEPlaintext> input,
                                                 const element::Type& type,
                                                 size_t count) const
{
    if (count != 1)
    {
        throw ngraph_error("Batching not enabled for SEAL in decode");
    }
    const string type_name = type.c_type_string();

    if (auto seal_input = dynamic_pointer_cast<SealPlaintextWrapper>(input))
    {
        if (type_name == "int64_t")
        {
            int64_t x = m_int_encoder->decode_int64(seal_input->m_plaintext);
            memcpy(output, &x, type.size());
        }
        else if (type_name == "float")
        {
            float x = m_frac_encoder->decode(seal_input->m_plaintext);
            memcpy(output, &x, type.size());
        }
        else
        {
            NGRAPH_INFO << "Unsupported element type in decode " << type_name;
            throw ngraph_error("Unsupported element type " + type_name);
        }
    }
    else
    {
        throw ngraph_error("HESealBackend::decode input is not seal plaintext");
    }
}

void runtime::he::he_seal::HESealBackend::encrypt(
    shared_ptr<runtime::he::HECiphertext>& output,
    const shared_ptr<runtime::he::HEPlaintext> input) const
{
    auto seal_output = dynamic_pointer_cast<runtime::he::SealCiphertextWrapper>(output);
    auto seal_input = dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(input);
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
    auto seal_output = dynamic_pointer_cast<runtime::he::SealPlaintextWrapper>(output);
    auto seal_input = dynamic_pointer_cast<runtime::he::SealCiphertextWrapper>(input);
    m_decryptor->decrypt(seal_input->m_ciphertext, seal_output->m_plaintext);
}

int runtime::he::he_seal::HESealBackend::noise_budget(
    const shared_ptr<seal::Ciphertext>& ciphertext) const
{
    return m_decryptor->invariant_noise_budget(*ciphertext);
}

void runtime::he::he_seal::HESealBackend::check_noise_budget(
    const vector<shared_ptr<runtime::he::HETensorView>>& tvs) const
{
    // Check noise budget
    NGRAPH_INFO << "Checking noise budget ";

    // Usually tvs.size() is very small (e.g. 1 for most ops), parallel the internal loops
    for (size_t i = 0; i < tvs.size(); ++i)
    {
        if (auto cipher_tv = dynamic_pointer_cast<HECipherTensorView>(tvs[i]))
        {
            size_t lowest_budget = numeric_limits<size_t>::max();

#pragma omp parallel for reduction(min : lowest_budget)
            for (size_t i = 0; i < cipher_tv->get_element_count(); ++i)
            {
                seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::New();
                shared_ptr<runtime::he::HECiphertext>& ciphertext = cipher_tv->get_element(i);

                if (auto seal_cipher_wrapper =
                        dynamic_pointer_cast<SealCiphertextWrapper>(ciphertext))
                {
                    int budget = m_decryptor->invariant_noise_budget(
                        seal_cipher_wrapper->m_ciphertext, pool);
                    if (budget <= 0)
                    {
                        NGRAPH_INFO << "Noise budget depleted";
                        throw ngraph_error("Noise budget depleted");
                    }
                    if (budget < lowest_budget)
                    {
                        lowest_budget = budget;
                    }
                }
            }
            NGRAPH_INFO << "Lowest noise budget " << lowest_budget;
        }
    }
    NGRAPH_INFO << "Done checking noise budget ";
}
