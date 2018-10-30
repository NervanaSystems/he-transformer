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
#include "he_tensor.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_parameter.hpp"
#include "seal/he_seal_util.hpp"

#include "seal/seal.h"

using namespace ngraph;
using namespace std;

const static runtime::he::he_seal::HESealParameter parse_seal_ckks_config_or_use_default()
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
            uint64_t security_level = js["security_level"];
            uint64_t evaluation_decomposition_bit_count = js["evaluation_decomposition_bit_count"];

            NGRAPH_INFO << "Using SEAL CKKS config for parameters: " << config_path;
            return runtime::he::he_seal::HESealParameter(scheme_name,
                                                         poly_modulus_degree,
                                                         security_level,
                                                         evaluation_decomposition_bit_count);
        }
        else
        {
            NGRAPH_INFO << "Using SEAL CKKS default parameters" << config_path;
            throw runtime_error("config_path is NULL");
        }
    }
    catch (const exception& e)
    {
        return runtime::he::he_seal::HESealParameter("HE:SEAL:CKKS", // scheme name
                                                     8192,           // poly_modulus_degree
                                                     128,            // security_level
                                                     60 // evaluation_decomposition_bit_count
                                                     );
    }
}

const static runtime::he::he_seal::HESealParameter default_seal_ckks_parameter =
    parse_seal_ckks_config_or_use_default();

runtime::he::he_seal::HESealCKKSBackend::HESealCKKSBackend()
    : runtime::he::he_seal::HESealCKKSBackend(
          make_shared<runtime::he::he_seal::HESealParameter>(default_seal_ckks_parameter))
{
}

runtime::he::he_seal::HESealCKKSBackend::HESealCKKSBackend(
    const shared_ptr<runtime::he::he_seal::HESealParameter>& sp)
{
    assert_valid_seal_ckks_parameter(sp);

    m_context = make_seal_context(sp);
    print_seal_context(*m_context);

    auto m_context_data = m_context->context_data();

    auto poly_modulus = m_context_data->parms().poly_modulus_degree();
    auto plain_modulus = m_context_data->parms().plain_modulus().value();

    // Keygen, encryptor and decryptor
    m_keygen = make_shared<seal::KeyGenerator>(m_context);
    m_relin_keys = make_shared<seal::RelinKeys>(
        m_keygen->relin_keys(sp->m_evaluation_decomposition_bit_count));
    m_public_key = make_shared<seal::PublicKey>(m_keygen->public_key());
    m_secret_key = make_shared<seal::SecretKey>(m_keygen->secret_key());
    m_encryptor = make_shared<seal::Encryptor>(m_context, *m_public_key);
    m_decryptor = make_shared<seal::Decryptor>(m_context, *m_secret_key);

    // Evaluator
    m_evaluator = make_shared<seal::Evaluator>(m_context);

    m_scale = static_cast<double>(m_context_data->parms().coeff_modulus().back().value());

    // Encoder
    m_ckks_encoder = make_shared<seal::CKKSEncoder>(m_context);

    // Plaintext constants
    shared_ptr<runtime::he::HEPlaintext> plaintext_neg1 = create_empty_plaintext();
    shared_ptr<runtime::he::HEPlaintext> plaintext_0 = create_empty_plaintext();
    shared_ptr<runtime::he::HEPlaintext> plaintext_1 = create_empty_plaintext();

    m_ckks_encoder->encode(
        -1,
        m_scale,
        dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(plaintext_neg1)
            ->m_plaintext);
    m_ckks_encoder->encode(
        0,
        m_scale,
        dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(plaintext_0)->m_plaintext);
    m_ckks_encoder->encode(
        1,
        m_scale,
        dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(plaintext_1)->m_plaintext);

    m_plaintext_map[-1] = plaintext_neg1;
    m_plaintext_map[0] = plaintext_0;
    m_plaintext_map[1] = plaintext_1;
}

extern "C" runtime::Backend* new_ckks_backend(const char* configuration_string)
{
    return new runtime::he::he_seal::HESealCKKSBackend();
}

shared_ptr<seal::SEALContext> runtime::he::he_seal::HESealCKKSBackend::make_seal_context(
    const shared_ptr<runtime::he::he_seal::HESealParameter> sp) const
{
    seal::EncryptionParameters parms =
        (sp->m_scheme_name == "HE:SEAL:CKKS"
             ? seal::scheme_type::CKKS
             : throw ngraph_error("Invalid scheme name \"" + sp->m_scheme_name + "\""));

    parms.set_poly_modulus_degree(sp->m_poly_modulus_degree);
    if (sp->m_security_level == 128)
    {
        // parms.set_coeff_modulus(seal::coeff_modulus_128(sp->m_poly_modulus_degree));

        // Slower, but increases multiplicative depth. Security remains valid, since 6*40 = 240 bits < 218 bits for default coeff.
        parms.set_coeff_modulus({
            seal::small_mods_40bit(0),
            seal::small_mods_40bit(1),
            seal::small_mods_40bit(2),
            seal::small_mods_40bit(3),
            seal::small_mods_40bit(4),
            seal::small_mods_40bit(5),
            seal::small_mods_40bit(6),
            seal::small_mods_40bit(7) // TODO: use fewer moduli to preserve security.
        });
    }
    else if (sp->m_security_level == 192)
    {
        parms.set_coeff_modulus(seal::coeff_modulus_192(sp->m_poly_modulus_degree));
    }
    else
    {
        throw ngraph_error("sp.security_level must be 128, 192");
    }
    return seal::SEALContext::Create(parms);
}

namespace
{
    static class HESealCKKSStaticInit
    {
    public:
        HESealCKKSStaticInit()
        {
            runtime::BackendManager::register_backend("HE:SEAL:CKKS", new_ckks_backend);
        }
        ~HESealCKKSStaticInit() {}
    } s_he_seal_ckks_static_init;
}

void runtime::he::he_seal::HESealCKKSBackend::assert_valid_seal_ckks_parameter(
    const shared_ptr<runtime::he::he_seal::HESealParameter>& sp) const
{
    assert_valid_seal_parameter(sp);
    if (sp->m_scheme_name != "HE:SEAL:CKKS")
    {
        throw ngraph_error("Invalid scheme name");
    }
}

shared_ptr<runtime::Tensor> runtime::he::he_seal::HESealCKKSBackend::create_batched_tensor(
    const element::Type& element_type, const Shape& shape)
{
    auto rc = make_shared<runtime::he::HECipherTensor>(
        element_type, shape, this, create_empty_ciphertext(), true);
    return static_pointer_cast<runtime::Tensor>(rc);
}

void runtime::he::he_seal::HESealCKKSBackend::encode(shared_ptr<runtime::he::HEPlaintext>& output,
                                                     const void* input,
                                                     const element::Type& type,
                                                     size_t count) const
{
    const string type_name = type.c_type_string();
    if (type_name == "float")
    {
        if (count == 1)
        {
            double value = (double)(*(float*)input);
            if (m_plaintext_map.find(value) != m_plaintext_map.end())
            {
                output = get_valued_plaintext(value);
            }
            else
            {
                m_ckks_encoder->encode(
                    value,
                    m_scale,
                    dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(output)
                        ->m_plaintext);
            }
        }
        else
        {
            vector<float> values{(float*)input, (float*)input + count};
            vector<double> double_values(values.begin(), values.end());
            m_ckks_encoder->encode(
                double_values,
                m_scale,
                dynamic_pointer_cast<runtime::he::he_seal::SealPlaintextWrapper>(output)
                    ->m_plaintext);
        }
    }
    else
    {
        NGRAPH_INFO << "Unsupported element type in decode " << type_name;
        throw ngraph_error("Unsupported element type " + type_name);
    }
}

void runtime::he::he_seal::HESealCKKSBackend::decode(
    void* output,
    const shared_ptr<runtime::he::HEPlaintext> input,
    const element::Type& type,
    size_t count) const
{
    const string type_name = type.c_type_string();

    if (count == 0)
    {
        throw ngraph_error("Decode called on 0 elements");
    }

    if (type_name == "float")
    {
        auto seal_input = dynamic_pointer_cast<SealPlaintextWrapper>(input);
        if (!seal_input)
        {
            throw ngraph_error("HESealCKKSBackend::decode input is not seal plaintext");
        }
        vector<double> xs(count, 0);
        m_ckks_encoder->decode(seal_input->m_plaintext, xs);
        vector<float> xs_float(xs.begin(), xs.end());

        memcpy(output, &xs_float[0], type.size() * count);
    }
    else
    {
        throw ngraph_error("Unsupported element type " + type_name);
    }
}
