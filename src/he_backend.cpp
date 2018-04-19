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

#include "he_backend.hpp"

using namespace ngraph;
using namespace std;

runtime::he::HEBackend::HEBackend()
{
    seal::EncryptionParameters parms;
    parms.set_poly_modulus("1x^2048 + 1");
    parms.set_coeff_modulus(seal::coeff_modulus_128(2048));
    parms.set_plain_modulus(1 << 8);
    m_context = make_shared<seal::SEALContext>(parms);
    m_int_encoder = make_shared<seal::IntegerEncoder>(m_context->plain_modulus());
    m_frac_encoder = make_shared<seal::FractionalEncoder>(m_context->plain_modulus(), m_context->poly_modulus(), 64, 32, 3);
    m_keygen = make_shared<seal::KeyGenerator>(*m_context);
    m_public_key = make_shared<seal::PublicKey>(m_keygen->public_key());
    m_secret_key = make_shared<seal::SecretKey>(m_keygen->secret_key());
    m_encryptor = make_shared<seal::Encryptor>(*m_context, *m_public_key);
    m_decryptor = make_shared<seal::Decryptor>(*m_context, *m_secret_key);
    m_evaluator = make_shared<seal::Evaluator>(*m_context);
}

runtime::he::HEBackend::HEBackend(seal::SEALContext& context)
    : m_context(make_shared<seal::SEALContext>(context))
    , m_int_encoder(make_shared<seal::IntegerEncoder>(m_context->plain_modulus()))
    , m_frac_encoder(make_shared<seal::FractionalEncoder>(m_context->plain_modulus(), m_context->poly_modulus(), 64, 32, 3))
    , m_keygen(make_shared<seal::KeyGenerator>(*m_context))
    , m_public_key(make_shared<seal::PublicKey>(m_keygen->public_key()))
    , m_secret_key(make_shared<seal::SecretKey>(m_keygen->secret_key()))
    , m_encryptor(make_shared<seal::Encryptor>(*m_context, *m_public_key))
    , m_decryptor(make_shared<seal::Decryptor>(*m_context, *m_secret_key))
    , m_evaluator(make_shared<seal::Evaluator>(*m_context))
{
}

runtime::he::HEBackend::~HEBackend()
{
}

shared_ptr<runtime::TensorView>
    runtime::he::HEBackend::create_tensor(const element::Type& element_type, const Shape& shape)
{
    throw ngraph_error("Unimplemented");
}

shared_ptr<runtime::TensorView> runtime::he::HEBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    throw ngraph_error("Unimplemented");
}

bool runtime::he::HEBackend::compile(std::shared_ptr<Function> func)
{
    throw ngraph_error("Unimplemented");
}

bool runtime::he::HEBackend::call(std::shared_ptr<Function> func,
                                  const vector<shared_ptr<runtime::TensorView>>& outputs,
                                  const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    throw ngraph_error("Unimplemented");
}

void runtime::he::HEBackend::remove_compiled_function(std::shared_ptr<Function> func)
{
    throw ngraph_error("Unimplemented");
}

void runtime::he::HEBackend::encode(seal::Plaintext* output, const void* input, const ngraph::element::Type& type)
{
    const std::string type_name = type.c_type_string();

    if (type_name == "int32_t")
    {
        *output = m_int_encoder->encode(*(int32_t*)input);
    }
    else if (type_name == "int64_t")
    {
        *output = m_int_encoder->encode(*(int64_t*)input);
    }
    else if (type_name == "uint32_t")
    {
        *output = m_int_encoder->encode(*(uint32_t*)input);
    }
    else if (type_name == "uint64_t")
    {
        *output = m_int_encoder->encode(*(uint64_t*)input);
    }
    else if (type_name == "float")
    {
        *output = m_frac_encoder->encode(*(float*)input);
    }
    else if (type_name == "double")
    {
        *output = m_frac_encoder->encode(*(double*)input);
    }
    else
    {
        throw ngraph_error("Type not supported");
    }
}

void runtime::he::HEBackend::decode(void* output, const seal::Plaintext& input, const ngraph::element::Type& type)
{
    const std::string type_name = type.c_type_string();

    if (type_name == "int32_t")
    {
        int32_t x = m_int_encoder->decode_int32(input);
        memcpy(output, &x, type.size());
    }
    else if (type_name == "int64_t")
    {
        int64_t x = m_int_encoder->decode_int64(input);
        memcpy(output, &x, type.size());
    }
    else if (type_name == "uint32_t")
    {
        uint32_t x =  m_int_encoder->decode_int64(input);
        memcpy(output, &x, type.size());
    }
    else if (type_name == "uint64_t")
    {
        uint64_t x =  m_int_encoder->decode_int64(input);
        memcpy(output, &x, type.size());
    }
    else if (type_name == "float")
    {
        float x = m_frac_encoder->decode(input);
        memcpy(output, &x, type.size());
    }
    else if (type_name == "double")
    {
        double x = m_frac_encoder->decode(input);
        memcpy(output, &x, type.size());
    }
    else
    {
        throw ngraph_error("Type not supported");
    }
}
