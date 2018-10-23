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
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_parameter.hpp"
#include "he_tensor.hpp"
#include "seal/bfv/he_seal_bfv_backend.hpp"
#include "ngraph/runtime/backend_manager.hpp"

#include "seal/seal.h"

using namespace ngraph;
using namespace std;

const static runtime::he::he_seal::HESealParameter parse_seal_bfv_config_or_use_default()
{
    try
    {
        const char* config_path = getenv("NGRAPH_HE_SEAL_BFV_CONFIG");
        if (config_path != nullptr)
        {
            // Read file to string
            ifstream f(config_path);
            stringstream ss;
            ss << f.rdbuf();
            string s = ss.str();

            // Parse json
            nlohmann::json js = nlohmann::json::parse(s);
            string scheme_name = js["scheme_name"];
            // assert(scheme_name == "BFV");
            uint64_t poly_modulus_degree = js["poly_modulus_degree"];
            uint64_t plain_modulus = js["plain_modulus"];
            uint64_t security_level = js["security_level"];
            uint64_t fractional_encoder_integer_coeff_count = js["fractional_encoder_integer_coeff_count"];
            uint64_t fractional_encoder_fraction_coeff_count = js["fractional_encoder_fraction_coeff_count"];
            uint64_t fractional_encoder_base = js["fractional_encoder_base"];
            uint64_t evaluation_decomposition_bit_count = js["evaluation_decomposition_bit_count"];

            NGRAPH_INFO << "Using SEAL BFV config for parameters: " << config_path;
            return runtime::he::he_seal::HESealParameter(scheme_name,
                                                poly_modulus_degree,
                                                plain_modulus,
                                                security_level,
                                                fractional_encoder_integer_coeff_count,
                                                fractional_encoder_fraction_coeff_count,
                                                fractional_encoder_base,
                                                evaluation_decomposition_bit_count);
        }
        else
        {
            NGRAPH_INFO << "Using SEAL BFV default parameters" << config_path;
            throw runtime_error("config_path is NULL");
        }
    }
    catch (const exception& e)
    {
        return runtime::he::he_seal::HESealParameter("BFV", // scheme name
                                            2048,      // poly_modulus_degree
                                            1 << 8, // plain_modulus
                                            128,       // security_level
                                            64,   // fractional_encoder_integer_coeff_count
                                            32,         // fractional_encoder_fraction_coeff_count
                                             2,           // fractional_encoder_base
                                            16            // evaluation_decomposition_bit_count
                                            );
    }
}

const static runtime::he::he_seal::HESealParameter default_seal_bfv_parameter =
    parse_seal_bfv_config_or_use_default();

runtime::he::he_seal::HESealBFVBackend::HESealBFVBackend()
    : runtime::he::he_seal::HESealBFVBackend(
          make_shared<runtime::he::he_seal::HESealParameter>(default_seal_bfv_parameter))
{
}


runtime::he::he_seal::HESealBFVBackend::HESealBFVBackend(
    const shared_ptr<runtime::he::he_seal::HESealParameter>& sp)
    : runtime::he::he_seal::HESealBackend::HESealBackend(sp)
{
    NGRAPH_INFO << "Creating HESealBFV Backend from sp";
   //  assert_valid_seal_parameter(sp);

    // Context
    NGRAPH_INFO << "Making seal context";
    m_context = make_seal_context(sp);
    NGRAPH_INFO << "Made seal context";
    auto m_context_data = m_context->context_data();

    // Encoders
    auto poly_modulus = m_context_data->parms().plain_modulus().value();
    auto plain_modulus = m_context_data->parms().plain_modulus().value();

    m_int_encoder = make_shared<seal::IntegerEncoder>(plain_modulus);
    NGRAPH_INFO << "Created m int encoder";
    m_frac_encoder =
        make_shared<seal::FractionalEncoder>(plain_modulus,
                                             poly_modulus,
                                             64, // TODO: add to parameter
                                             32,
                                             2);
}

// Hack to fix weak pointer error. Better is to remove all shared_from_this() from code.
// static runtime::Backend* s_seal_bfv_backend = nullptr;

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    /*if (s_seal_bfv_backend == nullptr)
    {
       s_seal_bfv_backend = new runtime::he::he_seal::HESealBFVBackend();
    }
    return s_seal_bfv_backend; */

    return new runtime::he::he_seal::HESealBFVBackend();
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    NGRAPH_INFO << "Deleting backend";
   //  delete backend;
}

namespace
{
    static class HESealBFVStaticInit
    {
    public:
        HESealBFVStaticInit() { runtime::BackendManager::register_backend("HESealBFV", new_backend); }
        ~HESealBFVStaticInit() {}
    } s_he_seal_bfv_static_init;
}

/* shared_ptr<runtime::Tensor>
    runtime::he::he_seal::HESealBFVBackend::create_tensor(const element::Type& element_type,
                                                       const Shape& shape)
{
    shared_ptr<HESealBFVBackend> he_seal_backend =
        dynamic_pointer_cast<runtime::he::he_seal::HESealBFVBackend>(shared_from_this());
    auto rc = make_shared<runtime::he::HECipherTensor>(element_type, shape, he_seal_backend);
    return static_pointer_cast<runtime::Tensor>(rc);
} */

shared_ptr<runtime::Tensor> runtime::he::he_seal::HESealBFVBackend::create_batched_tensor(
    const element::Type& element_type, const Shape& shape)
{
    throw ngraph_error("HESealBFVBackend::create_batched_tensor unimplemented");

}

/* shared_ptr<runtime::he::HECiphertext>
    runtime::he::he_seal::HESealBFVBackend::create_empty_ciphertext(size_t batch_size) const
{
    if (batch_size != 1)
    {
        throw ngraph_error("HESealBFVBackend::create_empty_ciphertext only supports batch size 1");
    }
    return make_shared<runtime::he::SealCiphertextWrapper>();
}
*/
shared_ptr<runtime::he::HEPlaintext> runtime::he::he_seal::HESealBFVBackend::create_valued_plaintext(
    float value, const element::Type& element_type) const
{
    const string type_name = element_type.c_type_string();
    shared_ptr<runtime::he::HEPlaintext> plaintext = create_empty_plaintext();
    if (auto plaintext_seal = dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(plaintext))
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
    runtime::he::he_seal::HESealBFVBackend::get_valued_plaintext(int64_t value,
                                                              const element::Type& element_type)
{
    throw ngraph_error("HESealBFVBackend::get_valued_plaintext unimplemented");
    /* const string type_name = element_type.c_type_string();
    unordered_set<int64_t> stored_plaintext_values{-1, 0, 1};
    if (stored_plaintext_values.find(value) == stored_plaintext_values.end())
    {
        throw ngraph_error("Value not stored in stored plaintext values");
    }
    if ((m_plaintext_map.find(type_name) == m_plaintext_map.end()) ||
        m_plaintext_map[type_name].find(value) == m_plaintext_map[type_name].end())
    {
        throw ngraph_error("Type or value not stored in m_plaintext_map");
    }
    return m_plaintext_map[type_name][value]; */
}
/*
shared_ptr<runtime::he::HEPlaintext>
    runtime::he::he_seal::HESealBFVBackend::create_empty_plaintext() const
{
    return make_shared<SealPlaintextWrapper>();
} */

shared_ptr<runtime::Tensor> runtime::he::he_seal::HESealBFVBackend::create_valued_tensor(
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

shared_ptr<runtime::he::HECiphertext> runtime::he::he_seal::HESealBFVBackend::create_valued_ciphertext(
    float value, const element::Type& element_type, size_t batch_size) const
{
    if (batch_size != 1)
    {
        throw ngraph_error("HESealBFVBackend::create_valued_ciphertext only supports batch size 1");
    }
    const string type_name = element_type.c_type_string();
    auto ciphertext =
        dynamic_pointer_cast<runtime::he::he_seal::SealCiphertextWrapper>(create_empty_ciphertext());
    if (ciphertext == nullptr)
    {
        throw ngraph_error("Ciphertext is not seal ciphertext in HESealBFVBackend::create_valued_ciphertext");
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

void runtime::he::he_seal::HESealBFVBackend::encode(shared_ptr<runtime::he::HEPlaintext>& output,
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
        NGRAPH_INFO << "Encoding int " << (*(int64_t*)input);
        NGRAPH_INFO << "(m_int_encoder == NULL?) " << (m_int_encoder == nullptr);
        output =
            make_shared<runtime::he::he_seal::SealPlaintextWrapper>(m_int_encoder->encode(*(int64_t*)input));
    }
    else if (type_name == "float")
    {
        NGRAPH_INFO << "Encoding float";
        output =
            make_shared<runtime::he::he_seal::SealPlaintextWrapper>(m_frac_encoder->encode(*(float*)input));
    }
    else
    {
        NGRAPH_INFO << "Unsupported element type in decode " << type_name;
        throw ngraph_error("Unsupported element type " + type_name);
    }
}

void runtime::he::he_seal::HESealBFVBackend::decode(void* output,
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
        throw ngraph_error("HESealBFVBackend::decode input is not seal plaintext");
    }
}

void runtime::he::he_seal::HESealBFVBackend::encrypt(
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
        throw ngraph_error("HESealBFVBackend::encrypt has non-seal ciphertexts");
    }
}

void runtime::he::he_seal::HESealBFVBackend::decrypt(
    shared_ptr<runtime::he::HEPlaintext>& output,
    const shared_ptr<runtime::he::HECiphertext> input) const
{
    auto seal_output = dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(output);
    auto seal_input = dynamic_pointer_cast<runtime::he::he_seal::SealCiphertextWrapper>(input);
    m_decryptor->decrypt(seal_input->m_ciphertext, seal_output->m_plaintext);
}

int runtime::he::he_seal::HESealBFVBackend::noise_budget(
    const shared_ptr<seal::Ciphertext>& ciphertext) const
{
    return m_decryptor->invariant_noise_budget(*ciphertext);
}

void runtime::he::he_seal::HESealBFVBackend::check_noise_budget(
    const vector<shared_ptr<runtime::he::HETensor>>& tvs) const
{
    // Check noise budget
    NGRAPH_INFO << "Checking noise budget ";

    // Usually tvs.size() is very small (e.g. 1 for most ops), parallel the internal loops
    for (size_t i = 0; i < tvs.size(); ++i)
    {
        if (auto cipher_tv = dynamic_pointer_cast<HECipherTensor>(tvs[i]))
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