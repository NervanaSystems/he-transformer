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
#include "seal/he_seal_util.hpp"

#include "seal/seal.h"

using namespace ngraph;
using namespace std;

const static runtime::he::he_seal::HESealParameter parse_seal_bfv_config_or_use_default()
{
    try
    {
        const char* config_path = getenv("NGRAPH_HE_SEAL_CONFIG");
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
        return runtime::he::he_seal::HESealParameter("HE:SEAL:BFV", // scheme name
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
{
    assert_valid_seal_bfv_parameter(sp);

    m_context = make_seal_context(sp);
    print_seal_context(*m_context);

    auto m_context_data = m_context->context_data();

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

    // Encoder
    m_frac_encoder =
        make_shared<seal::FractionalEncoder>(plain_modulus,
                                             poly_modulus,
                                             sp->m_fractional_encoder_integer_coeff_count,
                                             sp->m_fractional_encoder_fraction_coeff_count,
                                             sp->m_fractional_encoder_base);
}

extern "C" runtime::Backend* new_bfv_backend(const char* configuration_string)
{
    return new runtime::he::he_seal::HESealBFVBackend();
}

/* extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
} */

shared_ptr<seal::SEALContext> runtime::he::he_seal::HESealBFVBackend::make_seal_context(
    const shared_ptr<runtime::he::he_seal::HESealParameter> sp) const
{
    seal::EncryptionParameters parms = (sp->m_scheme_name == "HE:SEAL:BFV" ? seal::scheme_type::BFV :
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

    NGRAPH_INFO << "Setting plain mod to " << sp->m_plain_modulus;
    parms.set_plain_modulus(sp->m_plain_modulus);

    auto tmp =seal::SEALContext::Create(parms);

    NGRAPH_INFO << "Created SEALContext(parmz)";

    return  tmp;
}

namespace
{
    static class HESealBFVStaticInit
    {
    public:
        HESealBFVStaticInit() { runtime::BackendManager::register_backend("HE:SEAL:BFV", new_bfv_backend); }
        ~HESealBFVStaticInit() {}
    } s_he_seal_bfv_static_init;
}

void runtime::he::he_seal::HESealBFVBackend::assert_valid_seal_bfv_parameter(const shared_ptr<runtime::he::he_seal::HESealParameter>& sp) const
{
    assert_valid_seal_parameter(sp);
    if (sp->m_scheme_name != "HE:SEAL:BFV")
    {
        throw ngraph_error("Invalid scheme name");
    }
}

shared_ptr<runtime::Tensor> runtime::he::he_seal::HESealBFVBackend::create_batched_tensor(
    const element::Type& element_type, const Shape& shape)
{
    throw ngraph_error("HESealBFVBackend::create_batched_tensor unimplemented");

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

    if (type_name == "float")
    {
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
        if (type_name == "float")
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