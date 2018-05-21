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

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/function.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"

#include "he_backend.hpp"
#include "he_call_frame.hpp"
#include "he_cipher_tensor_view.hpp"
#include "he_plain_tensor_view.hpp"
#include "he_tensor_view.hpp"
#include "pass/insert_relinearize.hpp"
#include "he_seal_backend.hpp"
#include "he_seal_parameter.hpp"

using namespace ngraph;
using namespace std;

extern "C" bool create_backend()
{
    runtime::Backend::register_backend("HE_Seal", make_shared<runtime::he::HESealBackend>());
    return true;
}

static void print_seal_context(const seal::SEALContext& context)
{
    NGRAPH_INFO << endl
                << "/ Encryption parameters:" << endl
                << "| poly_modulus: " << context.poly_modulus().to_string() << endl
                // Print the size of the true (product) coefficient modulus
                << "| coeff_modulus size: " << context.total_coeff_modulus().significant_bit_count()
                << " bits" << endl
                << "| plain_modulus: " << context.plain_modulus().value() << endl
                << "\\ noise_standard_deviation: " << context.noise_standard_deviation();
}

runtime::he::HESealBackend::HESealBackend()
    : runtime::he::HESealBackend(make_shared<runtime::he::HESealParameter>(runtime::he::default_seal_parameter))
{
}

runtime::he::HESealBackend::HESealBackend(const shared_ptr<runtime::he::HEParameter> hep)
    : runtime::he::HESealBackend(make_shared<runtime::he::HESealParameter>(hep->m_poly_modulus, hep->m_plain_modulus))
{
    //shared_ptr<runtime::he::HESealParameter> sp = make_shared<runtime::he::HESealParameter>(hep->m_poly_modulus, hep->m_plain_modulus);
    //runtime::he::HESealBackend(sp);
}

runtime::he::HESealBackend::HESealBackend(const shared_ptr<runtime::he::HESealParameter> sp)
{
    assert_valid_seal_parameter(sp);

    // Context
    m_context = make_seal_context(sp);
    print_seal_context(*m_context);

    // Encoders
    m_int_encoder = make_shared<seal::IntegerEncoder>(m_context->plain_modulus());
    m_frac_encoder =
        make_shared<seal::FractionalEncoder>(m_context->plain_modulus(),
                                             m_context->poly_modulus(),
                                             sp->m_fractional_encoder_integer_coeff_count,
                                             sp->m_fractional_encoder_fraction_coeff_count,
                                             sp->m_fractional_encoder_base);

    // Keygen, encryptor and decryptor
    m_keygen = make_shared<seal::KeyGenerator>(*m_context);
    m_public_key = make_shared<seal::PublicKey>(m_keygen->public_key());
    m_secret_key = make_shared<seal::SecretKey>(m_keygen->secret_key());
    m_encryptor = make_shared<seal::Encryptor>(*m_context, *m_public_key);
    m_decryptor = make_shared<seal::Decryptor>(*m_context, *m_secret_key);

    // Evaluator
    seal::EvaluationKeys ev_key;
    m_keygen->generate_evaluation_keys(sp->m_evaluation_decomposition_bit_count, ev_key);
    m_ev_key = make_shared<seal::EvaluationKeys>(ev_key);
    m_evaluator = make_shared<seal::Evaluator>(*m_context);

    // Plaintext constants
    m_plaintext_num = plaintext_num{m_frac_encoder->encode(1),
                                    m_frac_encoder->encode(-1),
                                    m_int_encoder->encode(1),
                                    m_int_encoder->encode(-1)};
}

runtime::he::HESealBackend::~HESealBackend()
{
}

void runtime::he::HESealBackend::assert_valid_seal_parameter(const shared_ptr<runtime::he::HESealParameter> sp)
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

shared_ptr<seal::SEALContext> runtime::he::HESealBackend::make_seal_context(const shared_ptr<runtime::he::HESealParameter> sp)
{
     assert_valid_seal_parameter(sp);

     seal::EncryptionParameters parms;
     parms.set_poly_modulus("1x^" + to_string(sp->m_poly_modulus) + " + 1");
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
     return make_shared<seal::SEALContext>(parms);
}

shared_ptr<runtime::TensorView>
    runtime::he::HESealBackend::create_tensor(const element::Type& element_type, const Shape& shape)
{
    shared_ptr<HESealBackend> he_seal_backend = dynamic_pointer_cast<runtime::he::HESealBackend>(shared_from_this());
    auto rc = make_shared<runtime::he::HECipherTensorView>(element_type, shape, he_seal_backend);
    return static_pointer_cast<runtime::TensorView>(rc);
}

shared_ptr<runtime::TensorView> runtime::he::HESealBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    throw ngraph_error("HESeal create_tensor unimplemented");
}

shared_ptr<runtime::TensorView>
    runtime::he::HESealBackend::create_plain_tensor(const element::Type& element_type,
                                                const Shape& shape)
{
	shared_ptr<HESealBackend> he_seal_backend = dynamic_pointer_cast<runtime::he::HESealBackend>(shared_from_this());
    auto rc = make_shared<runtime::he::HEPlainTensorView>(element_type, shape, he_seal_backend);
    return static_pointer_cast<runtime::TensorView>(rc);
}

std::shared_ptr<runtime::he::HECiphertext> runtime::he::HESealBackend::create_valued_ciphertext(
    float value, const element::Type& element_type, const seal::MemoryPoolHandle& pool) const
{
	throw ngraph_error("create_valued_ciphertext unimplemented");
    // For Encryptor, we use the memory-pool version
    // For encoder, we'll need to initialize the Encoder object with memory-pool, so the default
    // non-memory-pool is used here.
    const string type_name = element_type.c_type_string();
    shared_ptr<he::HECiphertext> ciphertext = create_empty_ciphertext(pool);

    /* if (type_name == "float")
    {
        seal::Plaintext plaintext = m_frac_encoder->encode(value);
        m_encryptor->encrypt(plaintext, *ciphertext, pool);
    }
    else if (type_name == "int64_t")
    {
        seal::Plaintext plaintext = m_int_encoder->encode(static_cast<int64_t>(value));
        m_encryptor->encrypt(plaintext, *ciphertext, pool);
    }
    else
    {
        throw ngraph_error("Type not supported at create_ciphertext");
    } */
    return ciphertext;
}

std::shared_ptr<runtime::he::HECiphertext>
    runtime::he::HESealBackend::create_empty_ciphertext(const seal::MemoryPoolHandle& pool) const
{
    throw ngraph_error("HESealBackend::create_empty_ciphertext unimplemented");
    // return make_shared<he::HECiphertext>(m_context->parms(), pool);
}

std::shared_ptr<runtime::he::HEPlaintext> runtime::he::HESealBackend::create_valued_plaintext(
    float value, const element::Type& element_type, const seal::MemoryPoolHandle& pool) const
{
	throw ngraph_error("create_valued_plaintext unimplemented");
    // Optimize value == 0 to use memory-pool
    /* if (value == 0)
    {
        return make_shared<seal::Plaintext>(
            m_context->parms().poly_modulus().coeff_count(), 0, pool);
    } */

    // For encoder, we'll need to initialize the Encoder object with memory-pool, so the default
    // non-memory-pool is used here.
    /* const string type_name = element_type.c_type_string();
    std::shared_ptr<runtime::he::HEPlaintext> plaintext = create_empty_plaintext(pool);
    if (type_name == "float")
    {
        *plaintext = m_frac_encoder->encode(value);
    }
    else if (type_name == "int64_t")
    {
        *plaintext = m_int_encoder->encode(static_cast<int64_t>(value));
    }
    else
    {
        throw ngraph_error("Type not supported at create_ciphertext");
    }
    return plaintext; */
}

std::shared_ptr<runtime::he::HEPlaintext>
    runtime::he::HESealBackend::create_empty_plaintext(const seal::MemoryPoolHandle& pool) const
{
    throw ngraph_error("HESealBackend::create_empty_plaintext unimplemnented");
    // Return a memory-pooled version 0-initialized plaintext
    // It's fine to return a 0-valued plaintext when requesting for "empty"
    // return make_shared<runtime::he::HEPlaintext>(m_context->parms().poly_modulus().coeff_count(), 0, pool);
}

std::shared_ptr<runtime::he::HECiphertext>
    runtime::he::HESealBackend::create_valued_ciphertext(float value,
                                                     const element::Type& element_type) const
{
    throw ngraph_error("create_valued_ciphertext unimplemented in seal");
    /* const string type_name = element_type.c_type_string();
    shared_ptr<runtime::he::HECiphertext> ciphertext = create_empty_ciphertext();
    if (type_name == "float")
    {
        seal::Plaintext plaintext = m_frac_encoder->encode(value);
        m_encryptor->encrypt(plaintext, *ciphertext);
    }
    else if (type_name == "int64_t")
    {
        seal::Plaintext plaintext = m_int_encoder->encode(static_cast<int64_t>(value));
        m_encryptor->encrypt(plaintext, *ciphertext);
    }
    else
    {
        throw ngraph_error("Type not supported at create_ciphertext");
    }
    return ciphertext; */
}

std::shared_ptr<runtime::he::HECiphertext> runtime::he::HESealBackend::create_empty_ciphertext() const
{
    throw ngraph_error("HESealBackend::create_empty_ciphertext unimplemented");
    // return make_shared<runtime::he::HECiphertext>(m_context->parms());
}

std::shared_ptr<runtime::he::HEPlaintext>
    runtime::he::HESealBackend::create_valued_plaintext(float value,
                                                    const element::Type& element_type) const
{
    throw ngraph_error("create_valued_plaintext unimplemented in seal");
    /* const string type_name = element_type.c_type_string();
    std::shared_ptr<runtime::he::HEPlaintext> plaintext = create_empty_plaintext();
    if (type_name == "float")
    {
        *plaintext = m_frac_encoder->encode(value);
    }
    else if (type_name == "int64_t")
    {
        *plaintext = m_int_encoder->encode(static_cast<int64_t>(value));
    }
    else
    {
        throw ngraph_error("Type not supported at create_ciphertext");
    }
    return plaintext; */
}

std::shared_ptr<runtime::he::HEPlaintext> runtime::he::HESealBackend::create_empty_plaintext() const
{
    throw ngraph_error("create_empty_plaintext unimplemented in seal");
    // return make_shared<seal::Plaintext>(m_context->parms().poly_modulus().coeff_count(), 0);
}

shared_ptr<runtime::TensorView> runtime::he::HESealBackend::create_valued_tensor(
    float value, const element::Type& element_type, const Shape& shape)
{
    throw ngraph_error("create_valued_tensor unimplemented in seal");
    /* auto tensor = static_pointer_cast<HECipherTensorView>(create_tensor(element_type, shape));
    vector<shared_ptr<seal::Ciphertext>>& cipher_texts = tensor->get_elements();
#pragma omp parallel for
    for (size_t i = 0; i < cipher_texts.size(); ++i)
    {
        seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::New(false);
        cipher_texts[i] = create_valued_ciphertext(value, element_type, pool);
    }
    return tensor; */
}

shared_ptr<runtime::TensorView> runtime::he::HESealBackend::create_valued_plain_tensor(
    float value, const element::Type& element_type, const Shape& shape)
{
    throw ngraph_error("create_valued_tensor plain unimplemented in seal");
    /* auto tensor = static_pointer_cast<HEPlainTensorView>(create_plain_tensor(element_type, shape));
    vector<shared_ptr<seal::Plaintext>>& plain_texts = tensor->get_elements();
#pragma omp parallel for
    for (size_t i = 0; i < plain_texts.size(); ++i)
    {
        seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::New(false);
        plain_texts[i] = create_valued_plaintext(value, element_type, pool);
    }
    return tensor; */
}

bool runtime::he::HESealBackend::compile(shared_ptr<Function> func)
{
    if (m_function_map.count(func) == 0)
    {
        shared_ptr<HESealBackend> he_seal_backend = dynamic_pointer_cast<runtime::he::HESealBackend>(shared_from_this());
        shared_ptr<Function> cf_func = clone_function(*func);

        // Run passes
        ngraph::pass::Manager pass_manager;
        pass_manager
            .register_pass<ngraph::pass::AssignLayout<descriptor::layout::DenseTensorViewLayout>>();
        pass_manager.register_pass<runtime::he::pass::InsertRelinearize>();
        pass_manager.run_passes(cf_func);

        // Create call frame
        shared_ptr<HECallFrame> call_frame = make_shared<HECallFrame>(cf_func, he_seal_backend);

        m_function_map.insert({func, call_frame});
    }
    return true;
}

bool runtime::he::HESealBackend::call(shared_ptr<Function> func,
                                  const vector<shared_ptr<runtime::TensorView>>& outputs,
                                  const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    compile(func);
    m_function_map.at(func)->call(outputs, inputs);
    return true;
}

void runtime::he::HESealBackend::clear_function_instance()
{
    m_function_map.clear();
}

void runtime::he::HESealBackend::remove_compiled_function(shared_ptr<Function> func)
{
    throw ngraph_error("HESealBackend remove compile function unimplemented");
}

void runtime::he::HESealBackend::encode(runtime::he::HEPlaintext& output,
                                    const void* input,
                                    const element::Type& type)
{
    throw  ngraph_error("HESealBackend::encode unimplemented");
    /* const string type_name = type.c_type_string();

    if (type_name == "int64_t")
    {
        output = m_int_encoder->encode(*(int64_t*)input);
    }
    else if (type_name == "float")
    {
        output = m_frac_encoder->encode(*(float*)input);
    }
    else
    {
        NGRAPH_INFO << "Unsupported element type in decode " << type_name;
        throw ngraph_error("Unsupported element type " + type_name);
    } */
}

void runtime::he::HESealBackend::decode(void* output,
                                    const runtime::he::HEPlaintext& input,
                                    const element::Type& type)
{
    throw  ngraph_error("HESealBackend::encode unimplemented");
    /* const string type_name = type.c_type_string();

    if (type_name == "int64_t")
    {
        int64_t x = m_int_encoder->decode_int64(input);
        memcpy(output, &x, type.size());
    }
    else if (type_name == "float")
    {
        float x = m_frac_encoder->decode(input);
        memcpy(output, &x, type.size());
    }
    else
    {
        NGRAPH_INFO << "Unsupported element type in decode " << type_name;
        throw ngraph_error("Unsupported element type " + type_name);
    } */
}

void runtime::he::HESealBackend::encrypt(runtime::he::HECiphertext& output, const runtime::he::HEPlaintext& input)
{
    throw ngraph_error("HESealBackend::encrypt unimplemented");
    // m_encryptor->encrypt(input, output);
}

void runtime::he::HESealBackend::decrypt(runtime::he::HEPlaintext& output, const he::HECiphertext& input)
{
    throw ngraph_error("HESealBackend::dencrypt unimplemented");
    // m_decryptor->decrypt(input, output);
}

int runtime::he::HESealBackend::noise_budget(const shared_ptr<seal::Ciphertext>& ciphertext) const
{
    throw ngraph_error("HESealBackend::noise_budget unimplemented");
    return m_decryptor->invariant_noise_budget(*ciphertext);
}

/* void runtime::he::HESealBackend::check_noise_budget(
    const vector<shared_ptr<runtime::he::HETensorView>>& tvs) const
{
    throw ngraph_error("HESealBackend::check_noise_budget unimplemented");
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
                seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::New(false);
                shared_ptr<seal::Ciphertext>& ciphertext = cipher_tv->get_element(i);
                int budget = m_decryptor->invariant_noise_budget(*ciphertext, pool);
                if (budget < lowest_budget)
                {
                    lowest_budget = budget;
                }
                if (budget <= 0)
                {
                    throw ngraph_error("Noise budget depleted");
                } // TODO: break if this is too slow
            }
            NGRAPH_INFO << "Lowest Noise budget " << lowest_budget;
        }
    }
    NGRAPH_INFO << "Done checking noise budget ";
} */

void runtime::he::HESealBackend::enable_performance_data(shared_ptr<Function> func, bool enable)
{
    // Enabled by default
}

vector<runtime::PerformanceCounter>
    runtime::he::HESealBackend::get_performance_data(shared_ptr<Function> func) const
{
    return m_function_map.at(func)->get_performance_data();
}

void runtime::he::HESealBackend::visualize_function_after_pass(const shared_ptr<Function>& func,
                                                           const string& file_name)
{
    compile(func);
    auto cf = m_function_map.at(func);
    auto compiled_func = cf->get_compiled_function();
    NGRAPH_INFO << "Visualize graph to " << file_name;
    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::VisualizeTree>(file_name);
    pass_manager.run_passes(compiled_func);
}
