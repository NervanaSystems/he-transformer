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

#include "nlohmann/json.hpp"

#include "he_cipher_tensor.hpp"
#include "he_ckks_backend.hpp"
#include "he_ckks_parameter.hpp"
#include "he_plain_tensor.hpp"
#include "he_tensor.hpp"

using namespace ngraph;
using namespace std;

const static runtime::he::HEHeaanParameter parse_ckks_config_or_use_default()
{
    try
    {
        const char* config_path = std::getenv("NGRAPH_HE_HEAAN_CONFIG");
        if (config_path != nullptr)
        {
            // Read file to string
            std::ifstream f(config_path);
            std::stringstream ss;
            ss << f.rdbuf();
            std::string s = ss.str();

            // Parse json
            nlohmann::json js = nlohmann::json::parse(s);
            std::uint64_t log2_poly_modulus = js["log2_poly_modulus"];
            std::uint64_t log2_plain_modulus = js["log2_plain_modulus"];
            std::uint64_t log2_precision = js["log2_precision"];

            NGRAPH_INFO << "Using HEAAN config for parameters: " << config_path;
            return runtime::he::HEHeaanParameter(
                log2_poly_modulus, log2_plain_modulus, log2_precision);
        }
        else
        {
            NGRAPH_INFO << "Using HEAAN default parameters" << config_path;
            throw std::runtime_error("config_path is NULL");
        }
    }
    catch (const std::exception& e)
    {
        return runtime::he::HEHeaanParameter(13,  // m_log2_poly_modulus
                                             383, // m_log2_plain_modulus
                                             32   // m_log2_precision
                                             );
    }
}

const static runtime::he::HEHeaanParameter default_ckks_parameter =
    parse_ckks_config_or_use_default();

static void print_ckks_context(const ckks::Context& context)
{
    NGRAPH_INFO << endl
                << "/ Encryption parameters:" << endl
                << "| poly_modulus: "
                << "1x^" << context.N << " + 1" << endl
                << "| coeff_modulus: " << context.logQ << " bits" << endl
                << "\\ noise_standard_deviation: " << context.sigma;
}

runtime::he::he_ckks::HEHeaanBackend::HEHeaanBackend()
    : runtime::he::he_ckks::HEHeaanBackend(
          make_shared<runtime::he::HEHeaanParameter>(default_ckks_parameter))
{
}

runtime::he::he_ckks::HEHeaanBackend::HEHeaanBackend(
    const shared_ptr<runtime::he::HEHeaanParameter> hp)
{
    NGRAPH_INFO << "[HEAAN parameter]";
    NGRAPH_INFO << "hp.m_log2_poly_modulus: " << hp->m_log2_poly_modulus;
    NGRAPH_INFO << "hp.m_log2_plain_modulus: " << hp->m_log2_plain_modulus;
    NGRAPH_INFO << "hp.m_log2_precision: " << hp->m_log2_precision;

    assert_valid_ckks_parameter(hp);
    // Context
    m_context = make_shared<ckks::Context>(hp->m_log2_poly_modulus, hp->m_log2_plain_modulus);
    print_ckks_context(*m_context);

    m_log2_precision = (long)hp->m_log2_precision;

    // Secret Key
    m_secret_key = make_shared<ckks::SecretKey>(m_context->logN);

    // Scheme
    m_scheme = make_shared<ckks::Scheme>(*m_secret_key, *m_context);

    // Plaintext constants
    m_plaintext_map["float"][0] = create_valued_plaintext(0, element::f32);
    m_plaintext_map["float"][1] = create_valued_plaintext(1, element::f32);
    m_plaintext_map["float"][-1] = create_valued_plaintext(-1, element::f32);
    m_plaintext_map["int64_t"][0] = create_valued_plaintext(0, element::i64);
    m_plaintext_map["int64_t"][1] = create_valued_plaintext(1, element::i64);
    m_plaintext_map["int64_t"][-1] = create_valued_plaintext(-1, element::i64);

    // Ciphertext constants
    m_ciphertext_map["float"][0] = create_valued_ciphertext(0, element::f32);
    m_ciphertext_map["int64_t"][0] = create_valued_ciphertext(0, element::i64);
}

runtime::he::he_ckks::HEHeaanBackend::~HEHeaanBackend()
{
}

void runtime::he::he_ckks::HEHeaanBackend::assert_valid_ckks_parameter(
    const shared_ptr<runtime::he::HEHeaanParameter> hp) const
{
    static const int base = 2;
    static const int depth = 4; // TODO: find depth dynamically for computation

    double security =
        3.6 * (1 << hp->m_log2_poly_modulus) / (depth + hp->m_log2_plain_modulus) - 110.;
    // TODO: check this matches with https://bitbucket.org/malb/lwe-estimator
    // as claimed on slide 1 at https://github.com/kimandrik/HEAAN/blob/master/slide-HEAAN.pdf

    if (security < 128)
    {
        NGRAPH_WARN << "Security " << security << " too small for depth " << depth;
        // throw ngraph_error("Security inadequate"); // TODO: enable
    }
}

shared_ptr<runtime::Tensor>
    runtime::he::he_ckks::HEHeaanBackend::create_tensor(const element::Type& element_type,
                                                         const Shape& shape)
{
    shared_ptr<HEHeaanBackend> he_ckks_backend =
        dynamic_pointer_cast<runtime::he::he_ckks::HEHeaanBackend>(shared_from_this());
    auto rc = make_shared<runtime::he::HECipherTensor>(element_type, shape, he_ckks_backend);
    return static_pointer_cast<runtime::Tensor>(rc);
}

shared_ptr<runtime::Tensor> runtime::he::he_ckks::HEHeaanBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, const bool batched)
{
    shared_ptr<HEHeaanBackend> he_ckks_backend =
        dynamic_pointer_cast<runtime::he::he_ckks::HEHeaanBackend>(shared_from_this());

    auto rc = make_shared<runtime::he::HECipherTensor>(
        element_type, shape, he_ckks_backend, batched);
    return static_pointer_cast<runtime::Tensor>(rc);
}

shared_ptr<runtime::Tensor> runtime::he::he_ckks::HEHeaanBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    throw ngraph_error("HEHeaan create_tensor unimplemented");
}

shared_ptr<runtime::Tensor>
    runtime::he::he_ckks::HEHeaanBackend::create_plain_tensor(const element::Type& element_type,
                                                               const Shape& shape)
{
    shared_ptr<HEHeaanBackend> he_ckks_backend =
        dynamic_pointer_cast<runtime::he::he_ckks::HEHeaanBackend>(shared_from_this());
    auto rc = make_shared<runtime::he::HEPlainTensor>(element_type, shape, he_ckks_backend);
    return static_pointer_cast<runtime::Tensor>(rc);
}

shared_ptr<runtime::he::HECiphertext>
    runtime::he::he_ckks::HEHeaanBackend::create_valued_ciphertext(
        float value, const element::Type& element_type, size_t batch_size) const
{
    auto ciphertext = dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(
        create_empty_ciphertext(batch_size));

    if (batch_size == 1)
    {
        ciphertext->m_ciphertext =
            m_scheme->encryptSingle((double)value, get_precision(), m_context->logQ);
    }
    else
    {
        vector<double> values(batch_size, (double)value);
        ciphertext->m_ciphertext = m_scheme->encrypt(values, get_precision(), m_context->logQ);
    }
    return ciphertext;
}

shared_ptr<runtime::he::HECiphertext>& runtime::he::he_ckks::HEHeaanBackend::get_valued_ciphertext(
    int64_t value, const element::Type& element_type, size_t batch_size)
{
    if (batch_size != 1)
    {
        throw ngraph_error("HEHeaanBackend::get_valued_ciphertext supports only Batch size 1");
    }
    const string type_name = element_type.c_type_string();
    if ((m_ciphertext_map.find(type_name) == m_ciphertext_map.end()) ||
        (m_ciphertext_map[type_name].find(value) == m_ciphertext_map[type_name].end()))
    {
        throw ngraph_error("Type or value not stored in m_ciphertext_map");
    }
    return m_ciphertext_map[type_name][value];
}

shared_ptr<runtime::he::HECiphertext>
    runtime::he::he_ckks::HEHeaanBackend::create_empty_ciphertext(size_t batch_size) const
{
    return make_shared<runtime::he::HeaanCiphertextWrapper>(batch_size);
}

shared_ptr<runtime::he::HEPlaintext> runtime::he::he_ckks::HEHeaanBackend::create_valued_plaintext(
    float value, const element::Type& element_type) const
{
    auto plaintext =
        dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(create_empty_plaintext());

    plaintext->m_plaintexts = {value};
    return plaintext;
}

shared_ptr<runtime::he::HEPlaintext>
    runtime::he::he_ckks::HEHeaanBackend::get_valued_plaintext(int64_t value,
                                                                const element::Type& element_type)
{
    const string type_name = element_type.c_type_string();
    if ((m_plaintext_map.find(type_name) == m_plaintext_map.end()) ||
        m_plaintext_map[type_name].find(value) == m_plaintext_map[type_name].end())
    {
        throw ngraph_error("Type or value not stored in m_plaintext_map");
    }
    return m_plaintext_map[type_name][value];
}

shared_ptr<runtime::he::HEPlaintext>
    runtime::he::he_ckks::HEHeaanBackend::create_empty_plaintext() const
{
    return make_shared<HeaanPlaintextWrapper>();
}

shared_ptr<runtime::Tensor> runtime::he::he_ckks::HEHeaanBackend::create_valued_tensor(
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

shared_ptr<runtime::Tensor> runtime::he::he_ckks::HEHeaanBackend::create_valued_plain_tensor(
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

void runtime::he::he_ckks::HEHeaanBackend::encode(shared_ptr<runtime::he::HEPlaintext>& output,
                                                   const void* input,
                                                   const element::Type& type,
                                                   size_t count) const
{
    const string type_name = type.c_type_string();
    vector<double> input_dbl(count);

    if (type_name == "double")
    {
#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            input_dbl[i] = (double)((double*)input)[i];
        }
        output = make_shared<runtime::he::HeaanPlaintextWrapper>(input_dbl);
    }
    else if (type_name == "int64_t")
    {
#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            input_dbl[i] = (double)((int64_t*)input)[i];
        }
        output = make_shared<runtime::he::HeaanPlaintextWrapper>(input_dbl);
    }
    else if (type_name == "float")
    {
#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            input_dbl[i] = (double)((float*)input)[i];
        }
        output = make_shared<runtime::he::HeaanPlaintextWrapper>(input_dbl);
    }
    else
    {
        NGRAPH_INFO << "Unsupported element type in encode " << type_name;
        throw ngraph_error("Unsupported element type " + type_name);
    }
}

void runtime::he::he_ckks::HEHeaanBackend::decode(void* output,
                                                   const shared_ptr<runtime::he::HEPlaintext> input,
                                                   const element::Type& type,
                                                   size_t count) const
{
    const string type_name = type.c_type_string();

    if (auto ckks_input = dynamic_pointer_cast<HeaanPlaintextWrapper>(input))
    {
        if (type_name == "int64_t")
        {
#pragma omp parallel for
            for (size_t i = 0; i < count; ++i)
            {
                int64_t x = round(ckks_input->m_plaintexts[i]);
                memcpy((char*)output + i * type.size(), &x, type.size());
            }
        }
        else if (type_name == "float")
        {
#pragma omp parallel for
            for (size_t i = 0; i < count; ++i)
            {
                float x = ckks_input->m_plaintexts[i];
                memcpy((char*)output + i * type.size(), &x, type.size());
            }
        }
        else if (type_name == "double")
        {
#pragma omp parallel for
            for (size_t i = 0; i < count; ++i)
            {
                double x = ckks_input->m_plaintexts[i];
                memcpy((char*)output + i * type.size(), &x, type.size());
            }
        }
        else
        {
            NGRAPH_INFO << "Unsupported element type in decode " << type_name;
            throw ngraph_error("Unsupported element type " + type_name);
        }
    }
    else
    {
        throw ngraph_error("HEHeaanBackend::decode input is not ckks plaintext");
    }
}

void runtime::he::he_ckks::HEHeaanBackend::encrypt(
    shared_ptr<runtime::he::HECiphertext>& output,
    const shared_ptr<runtime::he::HEPlaintext> input) const
{
    auto ckks_output = dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(output);
    auto ckks_input = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(input);
    if (ckks_output != nullptr && ckks_input != nullptr)
    {
        if (ckks_input->m_plaintexts.size() == 1)
        {
            ckks_output->m_ciphertext = m_scheme->encryptSingle(
                ckks_input->m_plaintexts[0], m_log2_precision, m_context->logQ);
        }
        else
        {
            ckks_output->m_ciphertext =
                m_scheme->encrypt(ckks_input->m_plaintexts, m_log2_precision, m_context->logQ);
        }
        ckks_output->m_count = ckks_input->m_plaintexts.size();
    }
    else
    {
        throw ngraph_error("HEHeaanBackend::encrypt has non-ckks ciphertexts");
    }
}

void runtime::he::he_ckks::HEHeaanBackend::decrypt(
    shared_ptr<runtime::he::HEPlaintext>& output,
    const shared_ptr<runtime::he::HECiphertext> input) const
{
    auto ckks_output = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(output);
    auto ckks_input = dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(input);
    if (ckks_output != nullptr && ckks_input != nullptr)
    {
        size_t batch_count = ckks_input->m_count;
        if (batch_count == 1)
        {
            ckks_output->m_plaintexts = {
                m_scheme->decryptSingle(*m_secret_key, ckks_input->m_ciphertext).real()};
        }
        else
        {
            vector<complex<double>> ciphertexts =
                m_scheme->decrypt(*m_secret_key, ckks_input->m_ciphertext);
            vector<double> real_ciphertexts(batch_count);

            transform(ciphertexts.begin(),
                      ciphertexts.end(),
                      real_ciphertexts.begin(),
                      [](complex<double>& n) { return n.real(); });

            ckks_output->m_plaintexts = real_ciphertexts;
        }
    }
    else
    {
        throw ngraph_error("HEHeaanBackend::decrypt has non-ckks ciphertexts");
    }
}
