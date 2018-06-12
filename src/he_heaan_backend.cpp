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

#include "he_cipher_tensor_view.hpp"
#include "he_heaan_backend.hpp"
#include "he_heaan_parameter.hpp"
#include "he_plain_tensor_view.hpp"
#include "he_tensor_view.hpp"

using namespace ngraph;
using namespace std;

static void print_heaan_context(const heaan::Context& context)
{
    NGRAPH_INFO << endl
                << "/ Encryption parameters:" << endl
                << "| poly_modulus: "
                << "1x^" << context.N << " + 1" << endl
                << "| coeff_modulus: " << context.logQ << " bits" << endl
                << "\\ noise_standard_deviation: " << context.sigma;
}

runtime::he::he_heaan::HEHeaanBackend::HEHeaanBackend()
    : runtime::he::he_heaan::HEHeaanBackend(
          make_shared<runtime::he::HEHeaanParameter>(runtime::he::default_heaan_parameter))
{
}

runtime::he::he_heaan::HEHeaanBackend::HEHeaanBackend(
    const shared_ptr<runtime::he::HEHeaanParameter> hp)
{
    assert_valid_heaan_parameter(hp);
    // Context
    m_context = make_shared<heaan::Context>(hp->m_log2_poly_modulus, hp->m_log2_plain_modulus);
    print_heaan_context(*m_context);

    m_log2_precision = (long)hp->m_log2_precision;

    // Secret Key
    m_secret_key = make_shared<heaan::SecretKey>(m_context->logN);

    // Scheme
    m_scheme = make_shared<heaan::Scheme>(*m_secret_key, *m_context);

    // TODO: add plaintext constants as in SEAL backend

    NGRAPH_INFO << "Created Heaan backend";
}

runtime::he::he_heaan::HEHeaanBackend::~HEHeaanBackend()
{
}

void runtime::he::he_heaan::HEHeaanBackend::assert_valid_heaan_parameter(
    const shared_ptr<runtime::he::HEHeaanParameter> hp) const
{
    static const int base = 2;
    static const int depth = 4; // TODO: find depth dynamically for computation

    double security =
        3.6 * (1 << hp->m_log2_poly_modulus) / (depth + hp->m_log2_plain_modulus) - 110.;
    // TODO: check this matches with https://bitbucket.org/malb/lwe-estimator

    if (security < 128)
    {
        NGRAPH_WARN << "Security " << security << " too small for depth " << depth;
        // throw ngraph_error("Security inadequate"); // TODO: enable
    }
}

shared_ptr<runtime::TensorView>
    runtime::he::he_heaan::HEHeaanBackend::create_tensor(const element::Type& element_type,
                                                         const Shape& shape)
{
    shared_ptr<HEHeaanBackend> he_heaan_backend =
        dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(shared_from_this());
    auto rc = make_shared<runtime::he::HECipherTensorView>(element_type, shape, he_heaan_backend);
    return static_pointer_cast<runtime::TensorView>(rc);
}

shared_ptr<runtime::TensorView> runtime::he::he_heaan::HEHeaanBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, const bool batched)
{
    shared_ptr<HEHeaanBackend> he_heaan_backend =
        dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(shared_from_this());

    auto rc = make_shared<runtime::he::HECipherTensorView>(
        element_type, shape, he_heaan_backend, batched);
    return static_pointer_cast<runtime::TensorView>(rc);
}

shared_ptr<runtime::TensorView> runtime::he::he_heaan::HEHeaanBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    throw ngraph_error("HEHeaan create_tensor unimplemented");
}

shared_ptr<runtime::TensorView>
    runtime::he::he_heaan::HEHeaanBackend::create_plain_tensor(const element::Type& element_type,
                                                               const Shape& shape)
{
    shared_ptr<HEHeaanBackend> he_heaan_backend =
        dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(shared_from_this());
    auto rc = make_shared<runtime::he::HEPlainTensorView>(element_type, shape, he_heaan_backend);
    return static_pointer_cast<runtime::TensorView>(rc);
}

shared_ptr<runtime::he::HECiphertext>
    runtime::he::he_heaan::HEHeaanBackend::create_valued_ciphertext(
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

shared_ptr<runtime::he::HECiphertext>
    runtime::he::he_heaan::HEHeaanBackend::create_empty_ciphertext(size_t batch_size) const
{
    return make_shared<runtime::he::HeaanCiphertextWrapper>(batch_size);
}

shared_ptr<runtime::he::HEPlaintext> runtime::he::he_heaan::HEHeaanBackend::create_valued_plaintext(
    float value, const element::Type& element_type) const
{
    auto plaintext =
        dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(create_empty_plaintext());

    plaintext->m_plaintexts = {value};
    return plaintext;
}

shared_ptr<runtime::he::HEPlaintext>
    runtime::he::he_heaan::HEHeaanBackend::create_empty_plaintext() const
{
    return make_shared<HeaanPlaintextWrapper>();
}

shared_ptr<runtime::TensorView> runtime::he::he_heaan::HEHeaanBackend::create_valued_tensor(
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

shared_ptr<runtime::TensorView> runtime::he::he_heaan::HEHeaanBackend::create_valued_plain_tensor(
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

void runtime::he::he_heaan::HEHeaanBackend::encode(shared_ptr<runtime::he::HEPlaintext>& output,
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

void runtime::he::he_heaan::HEHeaanBackend::decode(void* output,
                                                   const shared_ptr<runtime::he::HEPlaintext> input,
                                                   const element::Type& type,
                                                   size_t count) const
{
    const string type_name = type.c_type_string();

    if (auto heaan_input = dynamic_pointer_cast<HeaanPlaintextWrapper>(input))
    {
        if (type_name == "int64_t")
        {
#pragma omp parallel for
            for (size_t i = 0; i < count; ++i)
            {
                int64_t x = round(heaan_input->m_plaintexts[i]);
                memcpy((char*)output + i * type.size(), &x, type.size());
            }
        }
        else if (type_name == "float")
        {
#pragma omp parallel for
            for (size_t i = 0; i < count; ++i)
            {
                float x = heaan_input->m_plaintexts[i];
                memcpy((char*)output + i * type.size(), &x, type.size());
            }
        }
        else if (type_name == "double")
        {
#pragma omp parallel for
            for (size_t i = 0; i < count; ++i)
            {
                double x = heaan_input->m_plaintexts[i];
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
        throw ngraph_error("HEHeaanBackend::decode input is not heaan plaintext");
    }
}

void runtime::he::he_heaan::HEHeaanBackend::encrypt(
    shared_ptr<runtime::he::HECiphertext>& output,
    const shared_ptr<runtime::he::HEPlaintext> input) const
{
    auto heaan_output = dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(output);
    auto heaan_input = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(input);
    if (heaan_output != nullptr && heaan_input != nullptr)
    {
        if (heaan_input->m_plaintexts.size() == 1)
        {
            heaan_output->m_ciphertext = m_scheme->encryptSingle(
                heaan_input->m_plaintexts[0], m_log2_precision, m_context->logQ);
        }
        else
        {
            heaan_output->m_ciphertext =
                m_scheme->encrypt(heaan_input->m_plaintexts, m_log2_precision, m_context->logQ);
        }
        heaan_output->m_count = heaan_input->m_plaintexts.size();
    }
    else
    {
        throw ngraph_error("HEHeaanBackend::encrypt has non-heaan ciphertexts");
    }
}

void runtime::he::he_heaan::HEHeaanBackend::decrypt(
    shared_ptr<runtime::he::HEPlaintext>& output,
    const shared_ptr<runtime::he::HECiphertext> input) const
{
    auto heaan_output = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(output);
    auto heaan_input = dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(input);
    if (heaan_output != nullptr && heaan_input != nullptr)
    {
        size_t batch_count = heaan_input->m_count;
        if (batch_count == 1)
        {
            heaan_output->m_plaintexts = {
                m_scheme->decryptSingle(*m_secret_key, heaan_input->m_ciphertext).real()};
        }
        else
        {
            vector<complex<double>> ciphertexts =
                m_scheme->decrypt(*m_secret_key, heaan_input->m_ciphertext);
            vector<double> real_ciphertexts(batch_count);

            transform(ciphertexts.begin(),
                      ciphertexts.end(),
                      real_ciphertexts.begin(),
                      [](complex<double>& n) { return n.real(); });

            heaan_output->m_plaintexts = real_ciphertexts;
        }
    }
    else
    {
        throw ngraph_error("HEHeaanBackend::decrypt has non-heaan ciphertexts");
    }
}
